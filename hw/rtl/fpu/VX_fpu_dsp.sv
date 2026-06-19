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

`ifdef VX_CFG_FPU_TYPE_DSP

module VX_fpu_dsp import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
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

    input wire [INST_FPU_BITS-1:0] op_type,
    input wire [INST_FMT_BITS-1:0] fmt,
    input wire [INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  dataa,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  datab,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  datac,
    output wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] result,

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

    localparam NUM_PES_FMA  = `UP(NUM_LANES / `VX_CFG_FMA_PE_RATIO);
    localparam NUM_PES_DIV  = `UP(NUM_LANES / `VX_CFG_FDIV_PE_RATIO);
    localparam NUM_PES_SQRT = `UP(NUM_LANES / `VX_CFG_FSQRT_PE_RATIO);
    localparam NUM_PES_CVT  = `UP(NUM_LANES / `VX_CFG_FCVT_PE_RATIO);
    localparam NUM_PES_NCP  = `UP(NUM_LANES / `VX_CFG_FNCP_PE_RATIO);
    localparam CVT_LATENCY  = (`VX_CFG_XLEN == 64) ? `VX_CFG_LATENCY_FCVT + 1 : `VX_CFG_LATENCY_FCVT;

    localparam REQ_DATAW = NUM_LANES + TAG_WIDTH + INST_FPU_BITS + INST_FMT_BITS + INST_FRM_BITS + 3 * (NUM_LANES * `VX_CFG_XLEN);
    localparam RSP_DATAW = (NUM_LANES * `VX_CFG_XLEN) + 1 + $bits(fflags_t) + TAG_WIDTH;

    wire [NUM_FPCORES-1:0] per_core_valid_in;
    wire [NUM_FPCORES-1:0][REQ_DATAW-1:0] per_core_data_in;
    wire [NUM_FPCORES-1:0] per_core_ready_in;

    wire [NUM_FPCORES-1:0][NUM_LANES-1:0] per_core_mask_in;
    wire [NUM_FPCORES-1:0][TAG_WIDTH-1:0] per_core_tag_in;
    wire [NUM_FPCORES-1:0][INST_FPU_BITS-1:0] per_core_op_type;
    wire [NUM_FPCORES-1:0][INST_FMT_BITS-1:0] per_core_fmt;
    wire [NUM_FPCORES-1:0][INST_FRM_BITS-1:0] per_core_frm;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] per_core_dataa;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] per_core_datab;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] per_core_datac;

    wire [NUM_FPCORES-1:0] per_core_valid_out;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] per_core_result;
    wire [NUM_FPCORES-1:0][TAG_WIDTH-1:0] per_core_tag_out;
    wire [NUM_FPCORES-1:0] per_core_has_fflags;
    fflags_t [NUM_FPCORES-1:0] per_core_fflags;
    wire [NUM_FPCORES-1:0] per_core_ready_out;

    // Decode fpu core type
    wire [FPCORES_BITS-1:0] core_select = (op_type == INST_FPU_F2F) ? FPCORES_BITS'(FPU_CVT)
                                                                    : FPCORES_BITS'(op_type[3:2]);

    VX_stream_switch #(
        .DATAW       (REQ_DATAW),
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NUM_FPCORES)
    ) req_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (core_select),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   ({mask_in, tag_in, fmt, frm, dataa, datab, datac, op_type}),
        .data_out  (per_core_data_in),
        .valid_out (per_core_valid_in),
        .ready_out (per_core_ready_in)
    );

    for (genvar i = 0; i < NUM_FPCORES; ++i) begin : g_per_core_data_in
        assign {
            per_core_mask_in[i],
            per_core_tag_in[i],
            per_core_fmt[i],
            per_core_frm[i],
            per_core_dataa[i],
            per_core_datab[i],
            per_core_datac[i],
            per_core_op_type[i]
        } = per_core_data_in[i];
    end

    // FMA core ///////////////////////////////////////////////////////////////

    if (1) begin : g_fma

        wire [NUM_LANES-1:0] mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] data_out;
        wire pe_enable;
        wire [NUM_PES_FMA-1:0] pe_mask_out;
        wire [NUM_PES_FMA-1:0][(3*`VX_CFG_XLEN)-1:0] pe_data_in;
        wire [INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS-1:0] pe_shared;
        wire [NUM_PES_FMA-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] pe_data_out;
        wire [NUM_LANES-1:0][(3*`VX_CFG_XLEN)-1:0] lane_data;

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_data
            assign lane_data[i] = {per_core_datac[FPU_FMA][i], per_core_datab[FPU_FMA][i], per_core_dataa[FPU_FMA][i]};
        end

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_FMA),
            .LATENCY        (`VX_CFG_LATENCY_FMA),
            .DATA_IN_WIDTH  (3*`VX_CFG_XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`VX_CFG_XLEN),
            .SHARED_WIDTH   (INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (per_core_valid_in[FPU_FMA]),
            .mask_in       (per_core_mask_in[FPU_FMA]),
            .data_in       (lane_data),
            .shared_in     ({per_core_op_type[FPU_FMA], per_core_fmt[FPU_FMA], per_core_frm[FPU_FMA]}),
            .tag_in        (per_core_tag_in[FPU_FMA]),
            .ready_in      (per_core_ready_in[FPU_FMA]),
            .pe_enable     (pe_enable),
            .pe_mask_out   (pe_mask_out),
            .pe_data_out   (pe_data_in),
            .pe_shared_out (pe_shared),
            .pe_data_in    (pe_data_out),
            .valid_out     (per_core_valid_out[FPU_FMA]),
            .mask_out      (mask_out),
            .data_out      (data_out),
            .tag_out       (per_core_tag_out[FPU_FMA]),
            .ready_out     (per_core_ready_out[FPU_FMA])
        );

        fflags_t [NUM_LANES-1:0] fflags_lanes;

        // Each branch writes a fully-boxed XLEN result (F64 fills the width; F32
        // is NaN-boxed): the vendor F32 IPs box their 32-bit output explicitly,
        // the native units self-box. The response path is a pass-through.
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result
            assign per_core_result[FPU_FMA][i] = data_out[i][0+:`VX_CFG_XLEN];
            assign fflags_lanes[i] = data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS];
        end

    `ifdef QUARTUS
        for (genvar i = 0; i < NUM_PES_FMA; ++i) begin : g_units
            wire [INST_FPU_BITS-1:0] op_pe  = pe_shared[INST_FRM_BITS + INST_FMT_BITS +: INST_FPU_BITS];
            wire [INST_FMT_BITS-1:0] fmt_pe = pe_shared[INST_FRM_BITS +: INST_FMT_BITS];
            wire is_madd_pe = op_pe[1];
            wire is_neg_pe  = op_pe[0];
            wire is_sub_pe  = fmt_pe[1];

            reg [31:0] a32, b32, c32;
            always @(*) begin
                if (is_madd_pe) begin
                    a32 = {is_neg_pe ^ pe_data_in[i][31], pe_data_in[i][0 +: 31]};
                    b32 = pe_data_in[i][`VX_CFG_XLEN +: 32];
                    c32 = {(is_neg_pe ^ is_sub_pe) ^ pe_data_in[i][2*`VX_CFG_XLEN + 31],
                           pe_data_in[i][2*`VX_CFG_XLEN +: 31]};
                end else begin
                    if (is_neg_pe) begin // MUL
                        a32 = pe_data_in[i][0 +: 32];
                        b32 = pe_data_in[i][`VX_CFG_XLEN +: 32];
                        c32 = '0;
                    end else begin // ADD/SUB
                        a32 = pe_data_in[i][0 +: 32];
                        b32 = 32'h3f800000; // 1.0f
                        c32 = {is_sub_pe ^ pe_data_in[i][`VX_CFG_XLEN + 31], pe_data_in[i][`VX_CFG_XLEN +: 31]};
                    end
                end
            end

            acl_fmadd fmadd (
                .clk    (clk),
                .areset (1'b0),
                .en     (pe_enable),
                .a      (a32),
                .b      (b32),
                .c      (c32),
                .q      (pe_data_out[i][0 +: 32])
            );
            if (`VX_CFG_XLEN > 32) begin : g_box
                assign pe_data_out[i][32 +: (`VX_CFG_XLEN-32)] = '1; // NaN-box F32
            end
            assign pe_data_out[i][`VX_CFG_XLEN +: `FP_FLAGS_BITS] = 'x;
        end
        `UNUSED_VAR (pe_mask_out)
        assign per_core_has_fflags[FPU_FMA] = 0;
        assign fflags_lanes = 'x;
    `elsif VIVADO
        for (genvar i = 0; i < NUM_PES_FMA; ++i) begin : g_units
            // xil_fma computes a*b+c, so the FMA-core opcodes must be mapped
            // onto that form before driving the IP (mirrors the QUARTUS
            // acl_fmadd path and VX_fma_unit's internal remap):
            //   MUL        : a*b + 0
            //   ADD/SUB    : a*1.0 (+/-) b
            //   MADD/NMADD : (+/-)a*b (+/-) c
            // Without this, ADD/SUB would incorrectly compute a*b+c.
            wire [INST_FPU_BITS-1:0] op_pe  = pe_shared[INST_FRM_BITS + INST_FMT_BITS +: INST_FPU_BITS];
            wire [INST_FMT_BITS-1:0] fmt_pe = pe_shared[INST_FRM_BITS +: INST_FMT_BITS];
            wire is_madd_pe = op_pe[1];
            wire is_neg_pe  = op_pe[0];
            wire is_sub_pe  = fmt_pe[1];

            reg [31:0] a32, b32, c32;
            always @(*) begin
                if (is_madd_pe) begin
                    a32 = {is_neg_pe ^ pe_data_in[i][31], pe_data_in[i][0 +: 31]};
                    b32 = pe_data_in[i][`VX_CFG_XLEN +: 32];
                    c32 = {(is_neg_pe ^ is_sub_pe) ^ pe_data_in[i][2*`VX_CFG_XLEN + 31],
                           pe_data_in[i][2*`VX_CFG_XLEN +: 31]};
                end else begin
                    if (is_neg_pe) begin // MUL
                        a32 = pe_data_in[i][0 +: 32];
                        b32 = pe_data_in[i][`VX_CFG_XLEN +: 32];
                        c32 = '0;
                    end else begin // ADD/SUB
                        a32 = pe_data_in[i][0 +: 32];
                        b32 = 32'h3f800000; // 1.0f
                        c32 = {is_sub_pe ^ pe_data_in[i][`VX_CFG_XLEN + 31], pe_data_in[i][`VX_CFG_XLEN +: 31]};
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
            if (`VX_CFG_XLEN > 32) begin : g_box
                assign pe_data_out[i][32 +: (`VX_CFG_XLEN-32)] = '1; // NaN-box F32
            end
                                                          // NV, DZ, OF, UF, NX
            assign pe_data_out[i][`VX_CFG_XLEN +: `FP_FLAGS_BITS] = {tuser[2], 1'b0, tuser[1], tuser[0], 1'b0};
        end
        `UNUSED_VAR (pe_mask_out)
        assign per_core_has_fflags[FPU_FMA] = 1;
    `else
        // Separate F32 / F64 FMA cores; fmt[0] selects the result.
        // pe_shared is input-aligned, so delay the selector by FMA latency to line
        // it up with the result emerging from the units (mirrors VX_fpu_std).
        wire is_d_in = (`VX_CFG_FLEN >= 64) & pe_shared[INST_FRM_BITS+0];
        wire is_d_fma;
        if (`VX_CFG_FLEN >= 64) begin : g_isd_pipe
            reg [`VX_CFG_LATENCY_FMA-1:0] is_d_sr;
            always @(posedge clk) begin
                if (reset) is_d_sr <= '0;
                else if (pe_enable) is_d_sr <= {is_d_sr[`VX_CFG_LATENCY_FMA-2:0], is_d_in};
            end
            assign is_d_fma = is_d_sr[`VX_CFG_LATENCY_FMA-1];
        end else begin : g_isd_s
            assign is_d_fma = 1'b0;
            `UNUSED_VAR (is_d_in)
            `UNUSED_VAR (is_d_fma)
        end
        for (genvar i = 0; i < NUM_PES_FMA; ++i) begin : g_units
            wire [31:0]               res32;
            wire [`FP_FLAGS_BITS-1:0] ff32;
            VX_fma_unit #(
                .LATENCY  (`VX_CFG_LATENCY_FMA),
                .MAN_BITS (23),
                .EXP_BITS (8),
                .USE_DSP  (`VX_CFG_USE_DSP)
            ) fma32 (
                .clk     (clk),
                .reset   (reset),
                .enable  (pe_enable),
                .mask    (pe_mask_out[i]),
                .op_type (pe_shared[INST_FRM_BITS+INST_FMT_BITS+:INST_FPU_BITS]),
                .fmt     (pe_shared[INST_FRM_BITS+:INST_FMT_BITS]),
                .frm     (pe_shared[0+:INST_FRM_BITS]),
                .dataa   (pe_data_in[i][0+:32]),
                .datab   (pe_data_in[i][`VX_CFG_XLEN+:32]),
                .datac   (pe_data_in[i][2*`VX_CFG_XLEN+:32]),
                .result  (res32),
                .fflags  (ff32)
            );
            if (`VX_CFG_FLEN >= 64) begin : g_fma_d
                wire [63:0]               res64;
                wire [`FP_FLAGS_BITS-1:0] ff64;
                VX_fma_unit #(
                    .LATENCY  (`VX_CFG_LATENCY_FMA),
                    .MAN_BITS (52),
                    .EXP_BITS (11),
                    .USE_DSP  (`VX_CFG_USE_DSP)
                ) fma64 (
                    .clk     (clk),
                    .reset   (reset),
                    .enable  (pe_enable),
                    .mask    (pe_mask_out[i]),
                    .op_type (pe_shared[INST_FRM_BITS+INST_FMT_BITS+:INST_FPU_BITS]),
                    .fmt     (pe_shared[INST_FRM_BITS+:INST_FMT_BITS]),
                    .frm     (pe_shared[0+:INST_FRM_BITS]),
                    .dataa   (pe_data_in[i][0+:64]),
                    .datab   (pe_data_in[i][`VX_CFG_XLEN+:64]),
                    .datac   (pe_data_in[i][2*`VX_CFG_XLEN+:64]),
                    .result  (res64),
                    .fflags  (ff64)
                );
                assign pe_data_out[i][0+:`VX_CFG_XLEN] = is_d_fma ? `VX_CFG_XLEN'(res64)
                                                                  : {{(`VX_CFG_XLEN-32){1'b1}}, res32};
                assign pe_data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS] = is_d_fma ? ff64 : ff32;
            end else begin : g_fma_s
                if (`VX_CFG_XLEN > 32) begin : g_box
                    assign pe_data_out[i][0+:`VX_CFG_XLEN] = {{(`VX_CFG_XLEN-32){1'b1}}, res32};
                end else begin : g_no_box
                    assign pe_data_out[i][0+:`VX_CFG_XLEN] = `VX_CFG_XLEN'(res32);
                end
                assign pe_data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS] = ff32;
            end
        end

        assign per_core_has_fflags[FPU_FMA] = 1;
    `endif

        fflags_t merged_fflags;
        `FPU_MERGE_FFLAGS(merged_fflags, fflags_lanes, mask_out, NUM_LANES);
        assign per_core_fflags[FPU_FMA] = merged_fflags;

    end

    // Div/Sqrt core //////////////////////////////////////////////////////////

    if (1) begin : g_fdivsqrt

        localparam PATH_REQ_DATAW = NUM_LANES + TAG_WIDTH + INST_FMT_BITS + INST_FRM_BITS + 2 * (NUM_LANES * `VX_CFG_XLEN);
        localparam PATH_RSP_DATAW = (NUM_LANES * `VX_CFG_XLEN) + 1 + $bits(fflags_t) + TAG_WIDTH;

        wire is_sqrt = per_core_op_type[FPU_DIVSQRT][0];

        wire [1:0] path_valid_in;
        wire [1:0][PATH_REQ_DATAW-1:0] path_data_in;
        wire [1:0] path_ready_in;

        wire [1:0][NUM_LANES-1:0] path_mask;
        wire [1:0][TAG_WIDTH-1:0] path_tag;
        wire [1:0][INST_FMT_BITS-1:0] path_fmt;
        wire [1:0][INST_FRM_BITS-1:0] path_frm;
        wire [1:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] path_dataa;
        wire [1:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] path_datab;

        `UNUSED_VAR (per_core_datac[FPU_DIVSQRT])
        `UNUSED_VAR (path_datab[1])

        VX_stream_switch #(
            .DATAW       (PATH_REQ_DATAW),
            .NUM_INPUTS  (1),
            .NUM_OUTPUTS (2),
            .OUT_BUF     (0)
        ) req_switch (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (is_sqrt),
            .valid_in  (per_core_valid_in[FPU_DIVSQRT]),
            .ready_in  (per_core_ready_in[FPU_DIVSQRT]),
            .data_in   ({per_core_mask_in[FPU_DIVSQRT], per_core_tag_in[FPU_DIVSQRT], per_core_fmt[FPU_DIVSQRT], per_core_frm[FPU_DIVSQRT], per_core_dataa[FPU_DIVSQRT], per_core_datab[FPU_DIVSQRT]}),
            .data_out  (path_data_in),
            .valid_out (path_valid_in),
            .ready_out (path_ready_in)
        );

        for (genvar i = 0; i < 2; ++i) begin : g_unpack
            assign {
                path_mask[i],
                path_tag[i],
                path_fmt[i],
                path_frm[i],
                path_dataa[i],
                path_datab[i]
            } = path_data_in[i];
        end

        wire [1:0] path_valid_out;
        wire [1:0][PATH_RSP_DATAW-1:0] path_data_out;
        wire [1:0] path_ready_out;

        // DIV path (index 0)

        wire [TAG_WIDTH-1:0] div_tag_out;
        wire [NUM_LANES-1:0] div_mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] div_data_out;
        wire div_pe_enable;
        wire [NUM_PES_DIV-1:0] div_pe_mask_out;
        wire [NUM_PES_DIV-1:0][(2*`VX_CFG_XLEN)-1:0] div_pe_data_in;
        wire [INST_FMT_BITS+INST_FRM_BITS-1:0] div_pe_shared;
        wire [NUM_PES_DIV-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] div_pe_data_out;
        wire [NUM_LANES-1:0][(2*`VX_CFG_XLEN)-1:0] div_lane_data;

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_div_lane_data
            assign div_lane_data[i] = {path_datab[0][i], path_dataa[0][i]};
        end

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_DIV),
            .LATENCY        (`VX_CFG_LATENCY_FDIV),
            .DATA_IN_WIDTH  (2*`VX_CFG_XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`VX_CFG_XLEN),
            .SHARED_WIDTH   (INST_FMT_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) div_pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (path_valid_in[0]),
            .mask_in       (path_mask[0]),
            .data_in       (div_lane_data),
            .shared_in     ({path_fmt[0], path_frm[0]}),
            .tag_in        (path_tag[0]),
            .ready_in      (path_ready_in[0]),
            .pe_enable     (div_pe_enable),
            .pe_mask_out   (div_pe_mask_out),
            .pe_data_out   (div_pe_data_in),
            .pe_shared_out (div_pe_shared),
            .pe_data_in    (div_pe_data_out),
            .valid_out     (path_valid_out[0]),
            .mask_out      (div_mask_out),
            .data_out      (div_data_out),
            .tag_out       (div_tag_out),
            .ready_out     (path_ready_out[0])
        );

        fflags_t [NUM_LANES-1:0] div_fflags_lanes;
        wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] div_result;

        // Pass-through: the F32 IPs box their output explicitly; the native
        // merged unit self-boxes (F64 fills the width, F32 NaN-boxed).
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_div_result
            assign div_result[i] = div_data_out[i][0+:`VX_CFG_XLEN];
            assign div_fflags_lanes[i] = div_data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS];
        end

        wire div_has_fflags;

    `ifdef QUARTUS
        for (genvar i = 0; i < NUM_PES_DIV; ++i) begin : g_div_units
            acl_fdiv fdiv (
                .clk    (clk),
                .areset (1'b0),
                .en     (div_pe_enable),
                .a      (div_pe_data_in[i][0 +: 32]),
                .b      (div_pe_data_in[i][`VX_CFG_XLEN +: 32]),
                .q      (div_pe_data_out[i][0 +: 32])
            );
            if (`VX_CFG_XLEN > 32) begin : g_box
                assign div_pe_data_out[i][32 +: (`VX_CFG_XLEN-32)] = '1; // NaN-box F32
            end
            assign div_pe_data_out[i][`VX_CFG_XLEN +: `FP_FLAGS_BITS] = 'x;
        end
        `UNUSED_VAR (div_pe_mask_out)
        assign div_has_fflags  = 0;
        assign div_fflags_lanes = 'x;
    `elsif VIVADO
        for (genvar i = 0; i < NUM_PES_DIV; ++i) begin : g_div_units
            wire [3:0] tuser;
            xil_fdiv fdiv (
                .aclk                (clk),
                .aclken              (div_pe_enable),
                .s_axis_a_tvalid     (1'b1),
                .s_axis_a_tdata      (div_pe_data_in[i][0 +: 32]),
                .s_axis_b_tvalid     (1'b1),
                .s_axis_b_tdata      (div_pe_data_in[i][`VX_CFG_XLEN +: 32]),
                `UNUSED_PIN (m_axis_result_tvalid),
                .m_axis_result_tdata (div_pe_data_out[i][0 +: 32]),
                .m_axis_result_tuser (tuser)
            );
            if (`VX_CFG_XLEN > 32) begin : g_box
                assign div_pe_data_out[i][32 +: (`VX_CFG_XLEN-32)] = '1; // NaN-box F32
            end
                                                          // NV, DZ, OF, UF, NX
            assign div_pe_data_out[i][`VX_CFG_XLEN +: `FP_FLAGS_BITS] = {tuser[2], tuser[3], tuser[1], tuser[0], 1'b0};
        end
        `UNUSED_VAR (div_pe_mask_out)
        assign div_has_fflags = 1;
    `else
        for (genvar i = 0; i < NUM_PES_DIV; ++i) begin : g_div_units
            if (`VX_CFG_XLEN > `VX_CFG_FLEN) begin : g_pad
                assign div_pe_data_out[i][`VX_CFG_FLEN +: (`VX_CFG_XLEN-`VX_CFG_FLEN)] = '0;
            end
            VX_fdivsqrt_unit #(
                .LATENCY (`VX_CFG_LATENCY_FDIV),
                .FLEN    (`VX_CFG_FLEN)
            ) fdiv_unit (
                .clk     (clk),
                .reset   (reset),
                .enable  (div_pe_enable),
                .mask    (div_pe_mask_out[i]),
                .fmt     (div_pe_shared[INST_FRM_BITS+:INST_FMT_BITS]),
                .frm     (div_pe_shared[0+:INST_FRM_BITS]),
                .dataa   (div_pe_data_in[i][0+:`VX_CFG_FLEN]),
                .datab   (div_pe_data_in[i][`VX_CFG_XLEN+:`VX_CFG_FLEN]),
                .is_sqrt (1'b0),
                .result  (div_pe_data_out[i][0+:`VX_CFG_FLEN]),
                .fflags  (div_pe_data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS])
            );
        end

        assign div_has_fflags = 1;
    `endif

        fflags_t div_merged_fflags;
        `FPU_MERGE_FFLAGS(div_merged_fflags, div_fflags_lanes, div_mask_out, NUM_LANES);
        assign path_data_out[0] = {div_result, div_has_fflags, div_merged_fflags, div_tag_out};

        // SQRT path (index 1)

        wire [TAG_WIDTH-1:0] sqrt_tag_out;
        wire [NUM_LANES-1:0] sqrt_mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] sqrt_data_out;
        wire sqrt_pe_enable;
        wire [NUM_PES_SQRT-1:0] sqrt_pe_mask_out;
        wire [NUM_PES_SQRT-1:0][`VX_CFG_XLEN-1:0] sqrt_pe_data_in;
        wire [INST_FMT_BITS+INST_FRM_BITS-1:0] sqrt_pe_shared;
        wire [NUM_PES_SQRT-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] sqrt_pe_data_out;

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_SQRT),
            .LATENCY        (`VX_CFG_LATENCY_FSQRT),
            .DATA_IN_WIDTH  (`VX_CFG_XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`VX_CFG_XLEN),
            .SHARED_WIDTH   (INST_FMT_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) sqrt_pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (path_valid_in[1]),
            .mask_in       (path_mask[1]),
            .data_in       (path_dataa[1]),
            .shared_in     ({path_fmt[1], path_frm[1]}),
            .tag_in        (path_tag[1]),
            .ready_in      (path_ready_in[1]),
            .pe_enable     (sqrt_pe_enable),
            .pe_mask_out   (sqrt_pe_mask_out),
            .pe_data_out   (sqrt_pe_data_in),
            .pe_shared_out (sqrt_pe_shared),
            .pe_data_in    (sqrt_pe_data_out),
            .valid_out     (path_valid_out[1]),
            .mask_out      (sqrt_mask_out),
            .data_out      (sqrt_data_out),
            .tag_out       (sqrt_tag_out),
            .ready_out     (path_ready_out[1])
        );

        fflags_t [NUM_LANES-1:0] sqrt_fflags_lanes;
        wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] sqrt_result;

        // Pass-through: F32 IP boxes explicitly; native merged unit self-boxes.
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_sqrt_result
            assign sqrt_result[i] = sqrt_data_out[i][0+:`VX_CFG_XLEN];
            assign sqrt_fflags_lanes[i] = sqrt_data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS];
        end

        wire sqrt_has_fflags;

    `ifdef QUARTUS
        for (genvar i = 0; i < NUM_PES_SQRT; ++i) begin : g_sqrt_units
            acl_fsqrt fsqrt (
                .clk    (clk),
                .areset (1'b0),
                .en     (sqrt_pe_enable),
                .a      (sqrt_pe_data_in[i][0 +: 32]),
                .q      (sqrt_pe_data_out[i][0 +: 32])
            );
            if (`VX_CFG_XLEN > 32) begin : g_box
                assign sqrt_pe_data_out[i][32 +: (`VX_CFG_XLEN-32)] = '1; // NaN-box F32
            end
            assign sqrt_pe_data_out[i][`VX_CFG_XLEN +: `FP_FLAGS_BITS] = 'x;
        end
        `UNUSED_VAR (sqrt_pe_mask_out)
        assign sqrt_has_fflags  = 0;
        assign sqrt_fflags_lanes = 'x;
    `elsif VIVADO
        for (genvar i = 0; i < NUM_PES_SQRT; ++i) begin : g_sqrt_units
            wire tuser;
            xil_fsqrt fsqrt (
                .aclk                (clk),
                .aclken              (sqrt_pe_enable),
                .s_axis_a_tvalid     (1'b1),
                .s_axis_a_tdata      (sqrt_pe_data_in[i][0 +: 32]),
                `UNUSED_PIN (m_axis_result_tvalid),
                .m_axis_result_tdata (sqrt_pe_data_out[i][0 +: 32]),
                .m_axis_result_tuser (tuser)
            );
            if (`VX_CFG_XLEN > 32) begin : g_box
                assign sqrt_pe_data_out[i][32 +: (`VX_CFG_XLEN-32)] = '1; // NaN-box F32
            end
                                                          // NV, DZ, OF, UF, NX
            assign sqrt_pe_data_out[i][`VX_CFG_XLEN +: `FP_FLAGS_BITS] = {tuser, 1'b0, 1'b0, 1'b0, 1'b0};
        end
        `UNUSED_VAR (sqrt_pe_mask_out)
        assign sqrt_has_fflags = 1;
    `else
        for (genvar i = 0; i < NUM_PES_SQRT; ++i) begin : g_sqrt_units
            if (`VX_CFG_XLEN > `VX_CFG_FLEN) begin : g_pad
                assign sqrt_pe_data_out[i][`VX_CFG_FLEN +: (`VX_CFG_XLEN-`VX_CFG_FLEN)] = '0;
            end
            VX_fdivsqrt_unit #(
                .LATENCY (`VX_CFG_LATENCY_FSQRT),
                .FLEN    (`VX_CFG_FLEN)
            ) fsqrt_unit (
                .clk     (clk),
                .reset   (reset),
                .enable  (sqrt_pe_enable),
                .mask    (sqrt_pe_mask_out[i]),
                .fmt     (sqrt_pe_shared[INST_FRM_BITS+:INST_FMT_BITS]),
                .frm     (sqrt_pe_shared[0+:INST_FRM_BITS]),
                .dataa   (sqrt_pe_data_in[i][0+:`VX_CFG_FLEN]),
                .datab   (`VX_CFG_FLEN'b0),
                .is_sqrt (1'b1),
                .result  (sqrt_pe_data_out[i][0+:`VX_CFG_FLEN]),
                .fflags  (sqrt_pe_data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS])
            );
        end

        assign sqrt_has_fflags = 1;
    `endif

        fflags_t sqrt_merged_fflags;
        `FPU_MERGE_FFLAGS(sqrt_merged_fflags, sqrt_fflags_lanes, sqrt_mask_out, NUM_LANES);
        assign path_data_out[1] = {sqrt_result, sqrt_has_fflags, sqrt_merged_fflags, sqrt_tag_out};

        VX_stream_arb #(
            .NUM_INPUTS (2),
            .DATAW      (PATH_RSP_DATAW),
            .ARBITER    ("P"),
            .OUT_BUF    (0)
        ) rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (path_valid_out),
            .ready_in  (path_ready_out),
            .data_in   (path_data_out),
            .data_out  ({per_core_result[FPU_DIVSQRT], per_core_has_fflags[FPU_DIVSQRT], per_core_fflags[FPU_DIVSQRT], per_core_tag_out[FPU_DIVSQRT]}),
            .valid_out (per_core_valid_out[FPU_DIVSQRT]),
            .ready_out (per_core_ready_out[FPU_DIVSQRT]),
            `UNUSED_PIN (sel_out)
        );

    end

    // CVT core ///////////////////////////////////////////////////////////////

    if (1) begin : g_cvt

        wire [NUM_LANES-1:0] mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] data_out;
        wire pe_enable;
        wire [NUM_PES_CVT-1:0] pe_mask_out;
        wire [NUM_PES_CVT-1:0][`VX_CFG_XLEN-1:0] pe_data_in;
        wire [INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS-1:0] pe_shared;
        wire [NUM_PES_CVT-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] pe_data_out;

        `UNUSED_VAR ({per_core_datab[FPU_CVT], per_core_datac[FPU_CVT]})

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_CVT),
            .LATENCY        (CVT_LATENCY),
            .DATA_IN_WIDTH  (`VX_CFG_XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`VX_CFG_XLEN),
            .SHARED_WIDTH   (INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (per_core_valid_in[FPU_CVT]),
            .mask_in       (per_core_mask_in[FPU_CVT]),
            .data_in       (per_core_dataa[FPU_CVT]),
            .shared_in     ({per_core_op_type[FPU_CVT], per_core_fmt[FPU_CVT], per_core_frm[FPU_CVT]}),
            .tag_in        (per_core_tag_in[FPU_CVT]),
            .ready_in      (per_core_ready_in[FPU_CVT]),
            .pe_enable     (pe_enable),
            .pe_mask_out   (pe_mask_out),
            .pe_data_out   (pe_data_in),
            .pe_shared_out (pe_shared),
            .pe_data_in    (pe_data_out),
            .valid_out     (per_core_valid_out[FPU_CVT]),
            .mask_out      (mask_out),
            .data_out      (data_out),
            .tag_out       (per_core_tag_out[FPU_CVT]),
            .ready_out     (per_core_ready_out[FPU_CVT])
        );

        fflags_t [NUM_LANES-1:0] fflags_lanes;

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result
            assign per_core_result[FPU_CVT][i] = data_out[i][0+:`VX_CFG_XLEN];
            assign fflags_lanes[i] = data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS];
        end

        for (genvar i = 0; i < NUM_PES_CVT; ++i) begin : g_units
            wire [INST_FRM_BITS-1:0] pe_frm = pe_shared[0+:INST_FRM_BITS];
            wire [INST_FMT_BITS-1:0] pe_fmt = pe_shared[INST_FRM_BITS+:INST_FMT_BITS];
            wire [INST_FPU_BITS-1:0] pe_op  = pe_shared[INST_FRM_BITS+INST_FMT_BITS+:INST_FPU_BITS];

            wire is_f2f    = (pe_op == INST_FPU_F2F);
            wire is_itof   = pe_op[1] & ~is_f2f;
            wire is_ftoi   = ~pe_op[1] & ~is_f2f;
            wire is_signed = ~pe_op[0];
            wire is_int64  = pe_fmt[1];
            wire src_fmt   = is_f2f ? ~pe_fmt[0] : pe_fmt[0]; // F2F: src is the other format
            wire dst_fmt   = pe_fmt[0];

            VX_fcvt_unit #(
                .LATENCY (CVT_LATENCY),
                .FLEN    (`VX_CFG_FLEN),
                .OUT_REG (1)
            ) fcvt_unit (
                .clk        (clk),
                .reset      (reset),
                .enable     (pe_enable),
                .mask       (pe_mask_out[i]),
                .frm        (pe_frm),
                .is_itof    (is_itof),
                .is_ftoi    (is_ftoi),
                .is_f2f     (is_f2f),
                .is_signed  (is_signed),
                .is_int64   (is_int64),
                .src_fmt    (src_fmt),
                .dst_fmt    (dst_fmt),
                .dataa      (pe_data_in[i][0+:`VX_CFG_XLEN]),
                .result     (pe_data_out[i][0+:`VX_CFG_XLEN]),
                .fflags     (pe_data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS])
            );
        end

        assign per_core_has_fflags[FPU_CVT] = 1;
        fflags_t merged_fflags;
        `FPU_MERGE_FFLAGS(merged_fflags, fflags_lanes, mask_out, NUM_LANES);
        assign per_core_fflags[FPU_CVT] = merged_fflags;

    end

    // NCP core ///////////////////////////////////////////////////////////////

    if (1) begin : g_ncp

        wire [NUM_LANES-1:0] mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] data_out;
        wire pe_enable;
        wire [NUM_PES_NCP-1:0] pe_mask_out;
        wire [NUM_PES_NCP-1:0][(2*`VX_CFG_XLEN)-1:0] pe_data_in;
        wire [INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS-1:0] pe_shared;
        wire [NUM_PES_NCP-1:0][(`FP_FLAGS_BITS+`VX_CFG_XLEN)-1:0] pe_data_out;
        wire [NUM_LANES-1:0][(2*`VX_CFG_XLEN)-1:0] lane_data;

        `UNUSED_VAR (per_core_datac[FPU_NCP])

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_data
            assign lane_data[i] = {per_core_datab[FPU_NCP][i], per_core_dataa[FPU_NCP][i]};
        end

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_NCP),
            .LATENCY        (`VX_CFG_LATENCY_FNCP),
            .DATA_IN_WIDTH  (2*`VX_CFG_XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`VX_CFG_XLEN),
            .SHARED_WIDTH   (INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (per_core_valid_in[FPU_NCP]),
            .mask_in       (per_core_mask_in[FPU_NCP]),
            .data_in       (lane_data),
            .shared_in     ({per_core_op_type[FPU_NCP], per_core_fmt[FPU_NCP], per_core_frm[FPU_NCP]}),
            .tag_in        (per_core_tag_in[FPU_NCP]),
            .ready_in      (per_core_ready_in[FPU_NCP]),
            .pe_enable     (pe_enable),
            .pe_mask_out   (pe_mask_out),
            .pe_data_out   (pe_data_in),
            .pe_shared_out (pe_shared),
            .pe_data_in    (pe_data_out),
            .valid_out     (per_core_valid_out[FPU_NCP]),
            .mask_out      (mask_out),
            .data_out      (data_out),
            .tag_out       (per_core_tag_out[FPU_NCP]),
            .ready_out     (per_core_ready_out[FPU_NCP])
        );

        fflags_t [NUM_LANES-1:0] fflags_lanes;

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result
            assign per_core_result[FPU_NCP][i] = data_out[i][0+:`VX_CFG_XLEN];
            assign fflags_lanes[i] = data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS];
        end

        for (genvar i = 0; i < NUM_PES_NCP; ++i) begin : g_units
            VX_fncp_unit #(
                .LATENCY (`VX_CFG_LATENCY_FNCP),
                .FLEN    (`VX_CFG_FLEN),
                .OUT_REG (1)
            ) fncp_unit (
                .clk     (clk),
                .reset   (reset),
                .enable  (pe_enable),
                .mask    (pe_mask_out[i]),
                .frm     (pe_shared[0+:INST_FRM_BITS]),
                .fmt     (pe_shared[INST_FRM_BITS+:INST_FMT_BITS]),
                .op_type (pe_shared[INST_FRM_BITS+INST_FMT_BITS+:INST_FPU_BITS]),
                .dataa   (pe_data_in[i][0+:`VX_CFG_FLEN]),
                .datab   (pe_data_in[i][`VX_CFG_XLEN+:`VX_CFG_FLEN]),
                .result  (pe_data_out[i][0+:`VX_CFG_XLEN]),
                .fflags  (pe_data_out[i][`VX_CFG_XLEN+:`FP_FLAGS_BITS])
            );
        end

        assign per_core_has_fflags[FPU_NCP] = 1;
        fflags_t merged_fflags;
        `FPU_MERGE_FFLAGS(merged_fflags, fflags_lanes, mask_out, NUM_LANES);
        assign per_core_fflags[FPU_NCP] = merged_fflags;

    end

    ///////////////////////////////////////////////////////////////////////////

    reg [NUM_FPCORES-1:0][RSP_DATAW-1:0] per_core_data_out;

    always @(*) begin
        for (integer i = 0; i < NUM_FPCORES; ++i) begin
            per_core_data_out[i] = {
                per_core_result[i],
                per_core_has_fflags[i],
                per_core_fflags[i],
                per_core_tag_out[i]
            };
        end
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_FPCORES),
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

endmodule

`endif

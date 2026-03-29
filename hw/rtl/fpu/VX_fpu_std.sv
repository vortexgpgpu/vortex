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

`ifdef FPU_TYPE_STD

module VX_fpu_std import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
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

    localparam NUM_PES_FMA  = `UP(NUM_LANES / `FMA_PE_RATIO);
    localparam NUM_PES_DIV  = `UP(NUM_LANES / `FDIV_PE_RATIO);
    localparam NUM_PES_CVT  = `UP(NUM_LANES / `FCVT_PE_RATIO);
    localparam NUM_PES_NCP  = `UP(NUM_LANES / `FNCP_PE_RATIO);

    localparam CVT_LATENCY = (`XLEN == 64) ? `LATENCY_FCVT + 1 : `LATENCY_FCVT;

    localparam REQ_DATAW = NUM_LANES + TAG_WIDTH + INST_FPU_BITS + INST_FMT_BITS + INST_FRM_BITS + 3 * (NUM_LANES * `XLEN);
    localparam RSP_DATAW = (NUM_LANES * `XLEN) + 1 + $bits(fflags_t) + TAG_WIDTH;

    // Per-core request signals
    wire [NUM_FPCORES-1:0] per_core_valid_in;
    wire [NUM_FPCORES-1:0][REQ_DATAW-1:0] per_core_data_in;
    wire [NUM_FPCORES-1:0] per_core_ready_in;

    wire [NUM_FPCORES-1:0][NUM_LANES-1:0] per_core_mask;
    wire [NUM_FPCORES-1:0][TAG_WIDTH-1:0] per_core_tag;
    wire [NUM_FPCORES-1:0][INST_FPU_BITS-1:0] per_core_op;
    wire [NUM_FPCORES-1:0][INST_FMT_BITS-1:0] per_core_fmt;
    wire [NUM_FPCORES-1:0][INST_FRM_BITS-1:0] per_core_frm;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][`XLEN-1:0] per_core_dataa;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][`XLEN-1:0] per_core_datab;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][`XLEN-1:0] per_core_datac;

    // Per-core response signals
    wire [NUM_FPCORES-1:0] per_core_valid_out;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][`XLEN-1:0] per_core_result;
    wire [NUM_FPCORES-1:0][TAG_WIDTH-1:0] per_core_tag_out;
    wire [NUM_FPCORES-1:0] per_core_has_fflags;
    fflags_t [NUM_FPCORES-1:0] per_core_fflags;
    wire [NUM_FPCORES-1:0] per_core_ready_out;

    // Decode FPU core selector
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

    for (genvar i = 0; i < NUM_FPCORES; ++i) begin : g_core_unpack
        assign {
            per_core_mask[i],
            per_core_tag[i],
            per_core_fmt[i],
            per_core_frm[i],
            per_core_dataa[i],
            per_core_datab[i],
            per_core_datac[i],
            per_core_op[i]
        } = per_core_data_in[i];
    end

    ///////////////////////////////////////////////////////////////////////////
    // FMA core
    ///////////////////////////////////////////////////////////////////////////

    begin : g_fma

        wire [NUM_LANES-1:0] mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] data_out;
        wire pe_enable;
        wire [NUM_PES_FMA-1:0] pe_mask_out;
        wire [NUM_PES_FMA-1:0][(3*`XLEN)-1:0] pe_data_in;
        wire [INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS-1:0] pe_shared;
        wire [NUM_PES_FMA-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] pe_data_out;
        wire [NUM_LANES-1:0][(3*`XLEN)-1:0] lane_data;

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_data
            assign lane_data[i] = {per_core_datac[FPU_FMA][i], per_core_datab[FPU_FMA][i], per_core_dataa[FPU_FMA][i]};
        end

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_FMA),
            .LATENCY        (`LATENCY_FMA),
            .DATA_IN_WIDTH  (3*`XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`XLEN),
            .SHARED_WIDTH   (INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (per_core_valid_in[FPU_FMA]),
            .mask_in       (per_core_mask[FPU_FMA]),
            .data_in       (lane_data),
            .shared_in     ({per_core_op[FPU_FMA], per_core_fmt[FPU_FMA], per_core_frm[FPU_FMA]}),
            .tag_in        (per_core_tag[FPU_FMA]),
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

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result
            if (`XLEN > 32) begin : g_nan_box
                assign per_core_result[FPU_FMA][i] = {32'hffffffff, data_out[i][0+:32]};
            end else begin : g_no_nan_box
                assign per_core_result[FPU_FMA][i] = data_out[i][0+:`XLEN];
            end
            assign fflags_lanes[i] = data_out[i][`XLEN+:`FP_FLAGS_BITS];
        end

        for (genvar i = 0; i < NUM_PES_FMA; ++i) begin : g_units
            VX_fma_unit #(
                .LATENCY (`LATENCY_FMA)
            ) fma_unit (
                .clk     (clk),
                .reset   (reset),
                .enable  (pe_enable),
                .mask    (pe_mask_out[i]),
                .op_type (pe_shared[INST_FRM_BITS+INST_FMT_BITS+:INST_FPU_BITS]),
                .fmt     (pe_shared[INST_FRM_BITS+:INST_FMT_BITS]),
                .frm     (pe_shared[0+:INST_FRM_BITS]),
                .dataa   (pe_data_in[i][0+:32]),
                .datab   (pe_data_in[i][`XLEN+:32]),
                .datac   (pe_data_in[i][2*`XLEN+:32]),
                .result  (pe_data_out[i][0+:32]),
                .fflags  (pe_data_out[i][`XLEN+:`FP_FLAGS_BITS])
            );
        end

        assign per_core_has_fflags[FPU_FMA] = 1;
        fflags_t merged_fflags;
        `FPU_MERGE_FFLAGS(merged_fflags, fflags_lanes, mask_out, NUM_LANES);
        assign per_core_fflags[FPU_FMA] = merged_fflags;

    end

    ///////////////////////////////////////////////////////////////////////////
    // DIVSQRT core
    ///////////////////////////////////////////////////////////////////////////

    begin : g_fdivsqrt

        wire is_sqrt = per_core_op[FPU_DIVSQRT][0];

        wire [NUM_LANES-1:0] mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] data_out;
        wire pe_enable;
        wire [NUM_PES_DIV-1:0] pe_mask_out;
        wire [NUM_PES_DIV-1:0][(2*`XLEN)-1:0] pe_data_in;
        wire [1+INST_FMT_BITS+INST_FRM_BITS-1:0] pe_shared;
        wire [NUM_PES_DIV-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] pe_data_out;
        wire [NUM_LANES-1:0][(2*`XLEN)-1:0] lane_data;

        `UNUSED_VAR (per_core_datac[FPU_DIVSQRT])

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_data
            assign lane_data[i] = {per_core_datab[FPU_DIVSQRT][i], per_core_dataa[FPU_DIVSQRT][i]};
        end

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_DIV),
            .LATENCY        (`LATENCY_FDIV),
            .DATA_IN_WIDTH  (2*`XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`XLEN),
            .SHARED_WIDTH   (1+INST_FMT_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (per_core_valid_in[FPU_DIVSQRT]),
            .mask_in       (per_core_mask[FPU_DIVSQRT]),
            .data_in       (lane_data),
            .shared_in     ({is_sqrt, per_core_fmt[FPU_DIVSQRT], per_core_frm[FPU_DIVSQRT]}),
            .tag_in        (per_core_tag[FPU_DIVSQRT]),
            .ready_in      (per_core_ready_in[FPU_DIVSQRT]),
            .pe_enable     (pe_enable),
            .pe_mask_out   (pe_mask_out),
            .pe_data_out   (pe_data_in),
            .pe_shared_out (pe_shared),
            .pe_data_in    (pe_data_out),
            .valid_out     (per_core_valid_out[FPU_DIVSQRT]),
            .mask_out      (mask_out),
            .data_out      (data_out),
            .tag_out       (per_core_tag_out[FPU_DIVSQRT]),
            .ready_out     (per_core_ready_out[FPU_DIVSQRT])
        );

        fflags_t [NUM_LANES-1:0] fflags_lanes;

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result
            if (`XLEN > 32) begin : g_nan_box
                assign per_core_result[FPU_DIVSQRT][i] = {32'hffffffff, data_out[i][0+:32]};
            end else begin : g_no_nan_box
                assign per_core_result[FPU_DIVSQRT][i] = data_out[i][0+:`XLEN];
            end
            assign fflags_lanes[i] = data_out[i][`XLEN+:`FP_FLAGS_BITS];
        end

        for (genvar i = 0; i < NUM_PES_DIV; ++i) begin : g_units
            VX_fdivsqrt_unit #(
                .LATENCY (`LATENCY_FDIV)
            ) fdiv_sqrt_unit (
                .clk     (clk),
                .reset   (reset),
                .enable  (pe_enable),
                .mask    (pe_mask_out[i]),
                .fmt     (pe_shared[INST_FRM_BITS+:INST_FMT_BITS]),
                .frm     (pe_shared[0+:INST_FRM_BITS]),
                .dataa   (pe_data_in[i][0+:32]),
                .datab   (pe_data_in[i][`XLEN+:32]),
                .is_sqrt (pe_shared[INST_FRM_BITS+INST_FMT_BITS]),
                .result  (pe_data_out[i][0+:32]),
                .fflags  (pe_data_out[i][`XLEN+:`FP_FLAGS_BITS])
            );
        end

        assign per_core_has_fflags[FPU_DIVSQRT] = 1;
        fflags_t merged_fflags;
        `FPU_MERGE_FFLAGS(merged_fflags, fflags_lanes, mask_out, NUM_LANES);
        assign per_core_fflags[FPU_DIVSQRT] = merged_fflags;

    end

    ///////////////////////////////////////////////////////////////////////////
    // CVT core
    ///////////////////////////////////////////////////////////////////////////

    begin : g_cvt

        wire [NUM_LANES-1:0] mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] data_out;
        wire pe_enable;
        wire [NUM_PES_CVT-1:0] pe_mask_out;
        wire [NUM_PES_CVT-1:0][`XLEN-1:0] pe_data_in;
        wire [INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS-1:0] pe_shared;
        wire [NUM_PES_CVT-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] pe_data_out;

        `UNUSED_VAR ({per_core_datab[FPU_CVT], per_core_datac[FPU_CVT]})

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_CVT),
            .LATENCY        (CVT_LATENCY),
            .DATA_IN_WIDTH  (`XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`XLEN),
            .SHARED_WIDTH   (INST_FPU_BITS+INST_FMT_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (per_core_valid_in[FPU_CVT]),
            .mask_in       (per_core_mask[FPU_CVT]),
            .data_in       (per_core_dataa[FPU_CVT]),
            .shared_in     ({per_core_op[FPU_CVT], per_core_fmt[FPU_CVT], per_core_frm[FPU_CVT]}),
            .tag_in        (per_core_tag[FPU_CVT]),
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
            assign per_core_result[FPU_CVT][i] = data_out[i][0+:`XLEN];
            assign fflags_lanes[i] = data_out[i][`XLEN+:`FP_FLAGS_BITS];
        end

        for (genvar i = 0; i < NUM_PES_CVT; ++i) begin : g_units
            wire [INST_FRM_BITS-1:0] pe_frm = pe_shared[0+:INST_FRM_BITS];
            wire [INST_FMT_BITS-1:0] pe_fmt = pe_shared[INST_FRM_BITS+:INST_FMT_BITS];
            wire [INST_FPU_BITS-1:0] pe_op  = pe_shared[INST_FRM_BITS+INST_FMT_BITS+:INST_FPU_BITS];

            `UNUSED_VAR (pe_op)
            `UNUSED_VAR (pe_fmt[0])

            wire is_itof   = pe_op[1];
            wire is_ftoi   = ~pe_op[1];
            wire is_signed = ~pe_op[0];
            wire is_dst_64 = is_ftoi ? pe_fmt[1] : 1'b0;
            wire is_src_64 = is_itof ? pe_fmt[1] : 1'b0;

            VX_fcvt_unit #(
                .LATENCY (CVT_LATENCY),
                .OUT_REG (1)
            ) fcvt_unit (
                .clk        (clk),
                .reset      (reset),
                .enable     (pe_enable),
                .mask       (pe_mask_out[i]),
                .frm        (pe_frm),
                .is_itof    (is_itof),
                .is_ftoi    (is_ftoi),
                .is_signed  (is_signed),
                .is_dst_64  (is_dst_64),
                .is_src_64  (is_src_64),
                .dataa      (pe_data_in[i][0+:`XLEN]),
                .result     (pe_data_out[i][0+:`XLEN]),
                .fflags     (pe_data_out[i][`XLEN+:`FP_FLAGS_BITS])
            );
        end

        assign per_core_has_fflags[FPU_CVT] = 1;
        fflags_t merged_fflags;
        `FPU_MERGE_FFLAGS(merged_fflags, fflags_lanes, mask_out, NUM_LANES);
        assign per_core_fflags[FPU_CVT] = merged_fflags;

    end

    ///////////////////////////////////////////////////////////////////////////
    // NCP core
    ///////////////////////////////////////////////////////////////////////////

    begin : g_ncp

        wire [NUM_LANES-1:0] mask_out;
        wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] data_out;
        wire pe_enable;
        wire [NUM_PES_NCP-1:0] pe_mask_out;
        wire [NUM_PES_NCP-1:0][(2*`XLEN)-1:0] pe_data_in;
        wire [INST_FPU_BITS+INST_FRM_BITS-1:0] pe_shared;
        wire [NUM_PES_NCP-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] pe_data_out;
        wire [NUM_LANES-1:0][(2*`XLEN)-1:0] lane_data;

        `UNUSED_VAR (per_core_datac[FPU_NCP])

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_data
            assign lane_data[i] = {per_core_datab[FPU_NCP][i], per_core_dataa[FPU_NCP][i]};
        end

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_PES_NCP),
            .LATENCY        (`LATENCY_FNCP),
            .DATA_IN_WIDTH  (2*`XLEN),
            .DATA_OUT_WIDTH (`FP_FLAGS_BITS+`XLEN),
            .SHARED_WIDTH   (INST_FPU_BITS+INST_FRM_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (per_core_valid_in[FPU_NCP]),
            .mask_in       (per_core_mask[FPU_NCP]),
            .data_in       (lane_data),
            .shared_in     ({per_core_op[FPU_NCP], per_core_frm[FPU_NCP]}),
            .tag_in        (per_core_tag[FPU_NCP]),
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
            assign per_core_result[FPU_NCP][i] = data_out[i][0+:`XLEN];
            assign fflags_lanes[i] = data_out[i][`XLEN+:`FP_FLAGS_BITS];
        end

        for (genvar i = 0; i < NUM_PES_NCP; ++i) begin : g_units
            VX_fncp_unit #(
                .LATENCY (`LATENCY_FNCP),
                .OUT_REG (1)
            ) fncp_unit (
                .clk     (clk),
                .reset   (reset),
                .enable  (pe_enable),
                .mask    (pe_mask_out[i]),
                .frm     (pe_shared[0+:INST_FRM_BITS]),
                .op_type (pe_shared[INST_FRM_BITS+:INST_FPU_BITS]),
                .dataa   (pe_data_in[i][0+:32]),
                .datab   (pe_data_in[i][`XLEN+:32]),
                .result  (pe_data_out[i][0+:`XLEN]),
                .fflags  (pe_data_out[i][`XLEN+:`FP_FLAGS_BITS])
            );
        end

        assign per_core_has_fflags[FPU_NCP] = 1;
        fflags_t merged_fflags;
        `FPU_MERGE_FFLAGS(merged_fflags, fflags_lanes, mask_out, NUM_LANES);
        assign per_core_fflags[FPU_NCP] = merged_fflags;

    end

    ///////////////////////////////////////////////////////////////////////////
    // Response arbitration
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

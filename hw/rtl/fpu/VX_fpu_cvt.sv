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

module VX_fpu_cvt import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter NUM_LANES = 5,
    parameter NUM_PES   = `UP(NUM_LANES / `FCVT_PE_RATIO),
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
    localparam DATAW = `XLEN + INST_FRM_BITS + INST_FPU_BITS + INST_FMT_BITS + INST_FRM_BITS;

    `UNUSED_VAR ({datab, datac})

    wire [NUM_LANES-1:0][DATAW-1:0] data_in;

    wire [NUM_LANES-1:0] mask_out;
    wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] data_out;
    fflags_t [NUM_LANES-1:0] fflags_out;

    wire pe_enable;
    wire [NUM_PES-1:0] pe_mask_out;
    `UNUSED_VAR (pe_mask_out)
    wire [NUM_PES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] pe_data_out;
    wire [NUM_PES-1:0][DATAW-1:0] pe_data_in;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_data_in
        assign data_in[i][0  +: `XLEN] = dataa[i];
        assign data_in[i][`XLEN +: INST_FPU_BITS] = op_type;
        assign data_in[i][`XLEN + INST_FPU_BITS +: INST_FMT_BITS] = fmt;
        assign data_in[i][`XLEN + INST_FPU_BITS + INST_FMT_BITS +: INST_FRM_BITS] = frm;
    end

    // XLEN=64 needs one extra pipeline stage to separate Stage0 (negate) from Stage1 (LZC).
    localparam CVT_LATENCY = (`XLEN == 64) ? `LATENCY_FCVT + 1 : `LATENCY_FCVT;

    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (CVT_LATENCY),
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
        assign result[i] = data_out[i][0 +: `XLEN];
        assign fflags_out[i] = data_out[i][`XLEN +: `FP_FLAGS_BITS];
    end

    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fcvt_units

        wire [INST_FPU_BITS-1:0] pe_op_type = pe_data_in[0][`XLEN +: INST_FPU_BITS];
        wire [INST_FMT_BITS-1:0] pe_fmt = pe_data_in[0][`XLEN + INST_FPU_BITS +: INST_FMT_BITS];
        wire [INST_FRM_BITS-1:0] pe_frm = pe_data_in[0][`XLEN + INST_FPU_BITS + INST_FMT_BITS +: INST_FRM_BITS];

        `UNUSED_VAR (pe_op_type)

        wire is_itof   = pe_op_type[1];
        wire is_ftoi   = ~pe_op_type[1];
        wire is_signed = ~pe_op_type[0];
        // F64 float removed: I2F always targets F32 (is_dst_64=0); F2I always from F32 (is_src_64=0).
        // fmt[1] selects 64-bit integer (I64); fmt[0] selected F64 float (now unused).
        wire is_dst_64 = is_ftoi ? pe_fmt[1] : 1'b0; // F2I→I64 if fmt[1]; I2F→F32 always
        wire is_src_64 = is_itof ? pe_fmt[1] : 1'b0; // I2F from I64 if fmt[1]; F2I from F32 always

        VX_fcvt_unit #(
            .LATENCY (CVT_LATENCY),
            .OUT_REG (1)
        ) fcvt_unit (
            .clk        (clk),
            .reset      (reset),
            .enable     (pe_enable),
            .frm        (pe_frm),
            .is_itof    (is_itof),
            .is_ftoi    (is_ftoi),
            .is_signed  (is_signed),
            .is_dst_64  (is_dst_64),
            .is_src_64  (is_src_64),
            .dataa      (pe_data_in[i][0 +: `XLEN]),
            .result     (pe_data_out[i][0 +: `XLEN]),
            .fflags     (pe_data_out[i][`XLEN +: `FP_FLAGS_BITS])
        );
    end

    assign has_fflags = 1;

    `FPU_MERGE_FFLAGS(fflags, fflags_out, mask_out, NUM_LANES);

endmodule

`endif

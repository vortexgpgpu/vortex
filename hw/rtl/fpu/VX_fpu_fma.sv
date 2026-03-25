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
    localparam DATAW = 3 * `XLEN;

    wire [NUM_LANES-1:0][DATAW-1:0] data_in;

    wire [NUM_LANES-1:0] mask_out;
    wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] data_out;
    wire [NUM_LANES-1:0][`FP_FLAGS_BITS-1:0] fflags_out;

    wire pe_enable;
    wire [NUM_PES-1:0][DATAW-1:0] pe_data_in;
    wire [INST_FPU_BITS + INST_FMT_BITS + INST_FRM_BITS - 1:0] pe_shared_in;
    wire [NUM_PES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] pe_data_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_data_in
        assign data_in[i] = {datac[i], datab[i], dataa[i]};
    end

    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (`LATENCY_FMA),
        .DATA_IN_WIDTH (DATAW),
        .DATA_OUT_WIDTH (`FP_FLAGS_BITS + `XLEN),
        .SHARED_WIDTH (INST_FPU_BITS + INST_FMT_BITS + INST_FRM_BITS),
        .TAG_WIDTH  (TAG_WIDTH),
        .PE_REG     (0),
        .OUT_BUF    (2)
    ) pe_serializer (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .mask_in    (mask_in),
        .data_in    (data_in),
        .shared_in ({op_type, fmt, frm}),
        .tag_in     (tag_in),
        .ready_in   (ready_in),
        .pe_enable  (pe_enable),
        `UNUSED_PIN (pe_mask_out),
        .pe_data_out(pe_data_in),
        .pe_shared_out(pe_shared_in),
        .pe_data_in (pe_data_out),
        .valid_out  (valid_out),
        .mask_out   (mask_out),
        .data_out   (data_out),
        .tag_out    (tag_out),
        .ready_out  (ready_out)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result
        // NaN-box F32 result in 64-bit register (upper 32 bits must be all 1s per RISC-V spec)
        assign result[i]     = (`XLEN > 32) ? {32'hffffffff, data_out[i][0 +: 32]} : data_out[i][0 +: `XLEN];
        assign fflags_out[i] = data_out[i][`XLEN +: `FP_FLAGS_BITS];
    end

    fflags_t [NUM_LANES-1:0] per_lane_fflags;

`ifdef QUARTUS

    // F32-only hardware path: derive sign-manipulation from op_type/fmt
    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fmas
        wire [INST_FPU_BITS-1:0] op_pe  = pe_shared_in[INST_FRM_BITS + INST_FMT_BITS +: INST_FPU_BITS];
        wire [INST_FMT_BITS-1:0] fmt_pe = pe_shared_in[INST_FRM_BITS +: INST_FMT_BITS];
        wire is_madd_pe = op_pe[1];
        wire is_neg_pe  = op_pe[0];
        wire is_sub_pe  = fmt_pe[1];

        reg [31:0] a32, b32, c32;
        always @(*) begin
            if (is_madd_pe) begin
                a32 = {is_neg_pe ^ pe_data_in[i][31], pe_data_in[i][0 +: 31]};
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

`else

    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fmas
        wire [INST_FRM_BITS-1:0] frm_pe = pe_shared_in[0 +: INST_FRM_BITS];
        wire [INST_FMT_BITS-1:0] fmt_pe = pe_shared_in[INST_FRM_BITS +: INST_FMT_BITS];
        wire [INST_FPU_BITS-1:0] op_pe  = pe_shared_in[INST_FRM_BITS + INST_FMT_BITS +: INST_FPU_BITS];
        VX_fma_unit fma_unit (
            .clk     (clk),
            .reset   (reset),
            .enable  (pe_enable),
            .op_type (op_pe),
            .fmt     (fmt_pe),
            .frm     (frm_pe),
            .dataa   (pe_data_in[i][0       +: 32]),
            .datab   (pe_data_in[i][`XLEN   +: 32]),
            .datac   (pe_data_in[i][2*`XLEN +: 32]),
            .result  (pe_data_out[i][0      +: 32]),
            .fflags  (pe_data_out[i][`XLEN  +: `FP_FLAGS_BITS])
        );
    end

    assign has_fflags = 1;
    assign per_lane_fflags = fflags_out;

`endif

`FPU_MERGE_FFLAGS(fflags, per_lane_fflags, mask_out, NUM_LANES);

endmodule

`endif

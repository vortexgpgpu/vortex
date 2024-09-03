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

module VX_fpu_cvt import VX_fpu_pkg::*; #(
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

    input wire [`INST_FRM_BITS-1:0] frm,

    input wire is_itof,
    input wire is_signed,

    input wire [NUM_LANES-1:0][31:0]  dataa,
    output wire [NUM_LANES-1:0][31:0] result,

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAG_WIDTH-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    localparam DATAW = 32 + `INST_FRM_BITS + 1 + 1;

    wire [NUM_LANES-1:0][DATAW-1:0] data_in;
    wire [NUM_LANES-1:0] mask_out;
    wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+32)-1:0] data_out;
    fflags_t [NUM_LANES-1:0] fflags_out;

    wire pe_enable;
    wire [NUM_PES-1:0][DATAW-1:0] pe_data_in;
    wire [NUM_PES-1:0][(`FP_FLAGS_BITS+32)-1:0] pe_data_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign data_in[i][0  +: 32] = dataa[i];
        assign data_in[i][32 +: `INST_FRM_BITS] = frm;
        assign data_in[i][32 + `INST_FRM_BITS +: 1] = is_itof;
        assign data_in[i][32 + `INST_FRM_BITS + 1 +: 1] = is_signed;
    end

    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (`LATENCY_FCVT),
        .DATA_IN_WIDTH(DATAW),
        .DATA_OUT_WIDTH(`FP_FLAGS_BITS + 32),
        .TAG_WIDTH  (NUM_LANES + TAG_WIDTH),
        .PE_REG     (0),
        .OUT_BUF    (2)
    ) pe_serializer (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .data_in    (data_in),
        .tag_in     ({mask_in, tag_in}),
        .ready_in   (ready_in),
        .pe_enable  (pe_enable),
        .pe_data_out(pe_data_in),
        .pe_data_in (pe_data_out),
        .valid_out  (valid_out),
        .data_out   (data_out),
        .tag_out    ({mask_out, tag_out}),
        .ready_out  (ready_out)
    );

    `UNUSED_VAR (pe_data_in)

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign result[i] = data_out[i][0 +: 32];
        assign fflags_out[i] = data_out[i][32 +: `FP_FLAGS_BITS];
    end

    for (genvar i = 0; i < NUM_PES; ++i) begin : fcvt_units
        VX_fcvt_unit #(
            .LATENCY (`LATENCY_FCVT),
            .OUT_REG (1)
        ) fcvt_unit (
            .clk        (clk),
            .reset      (reset),
            .enable     (pe_enable),
            .frm        (pe_data_in[0][32 +: `INST_FRM_BITS]),
            .is_itof    (pe_data_in[0][32 + `INST_FRM_BITS +: 1]),
            .is_signed  (pe_data_in[0][32 + `INST_FRM_BITS + 1 +: 1]),
            .dataa      (pe_data_in[i][0 +: 32]),
            .result     (pe_data_out[i][0 +: 32]),
            .fflags     (pe_data_out[i][32 +: `FP_FLAGS_BITS])
        );
    end

    assign has_fflags = 1;

    `FPU_MERGE_FFLAGS(fflags, fflags_out, mask_out, NUM_LANES);

endmodule

`endif

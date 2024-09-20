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

module VX_fpu_ncp import VX_fpu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter NUM_PES   = `UP(NUM_LANES / `FNCP_PE_RATIO),
    parameter TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    output wire ready_in,
    input wire  valid_in,

    input wire [NUM_LANES-1:0] mask_in,

    input wire [TAG_WIDTH-1:0] tag_in,

    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][31:0]  dataa,
    input wire [NUM_LANES-1:0][31:0]  datab,
    output wire [NUM_LANES-1:0][31:0] result,

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAG_WIDTH-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    `UNUSED_VAR (frm)

    wire [NUM_LANES-1:0][2*32-1:0] data_in;
    wire [NUM_LANES-1:0] mask_out;
    wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+32)-1:0] data_out;
    fflags_t [NUM_LANES-1:0] fflags_out;

    wire pe_enable;
    wire [NUM_PES-1:0][2*32-1:0] pe_data_in;
    wire [NUM_PES-1:0][(`FP_FLAGS_BITS+32)-1:0] pe_data_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign data_in[i][0  +: 32] = dataa[i];
        assign data_in[i][32 +: 32] = datab[i];
    end

    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (`LATENCY_FNCP),
        .DATA_IN_WIDTH(2*32),
        .DATA_OUT_WIDTH(`FP_FLAGS_BITS + 32),
        .TAG_WIDTH  (NUM_LANES + TAG_WIDTH),
        .PE_REG     (0),
        .OUT_BUF    (((NUM_LANES / NUM_PES) > 2) ? 1 : 0)
    ) pe_serializer (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .data_in    (data_in),
        .tag_in     ({mask_in, tag_in}),
        .ready_in   (ready_in),
        .pe_enable  (pe_enable),
        .pe_data_in (pe_data_in),
        .pe_data_out(pe_data_out),
        .valid_out  (valid_out),
        .data_out   (data_out),
        .tag_out    ({mask_out, tag_out}),
        .ready_out  (ready_out)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign result[i] = data_out[i][0 +: 32];
        assign fflags_out[i] = data_out[i][32 +: `FP_FLAGS_BITS];
    end

    for (genvar i = 0; i < NUM_PES; ++i) begin
        VX_fncp_unit #(
            .LATENCY (`LATENCY_FNCP)
        ) fncp_unit (
            .clk        (clk),
            .reset      (reset),
            .enable     (pe_enable),
            .frm        (frm),
            .op_type    (op_type),
            .dataa      (pe_data_in[i][0 +: 32]),
            .datab      (pe_data_in[i][32 +: 32]),
            .result     (pe_data_out[i][0 +: 32]),
            .fflags     (pe_data_out[i][32 +: `FP_FLAGS_BITS])
        );
    end

    assign has_fflags = 1;

    `FPU_MERGE_FFLAGS(fflags, fflags_out, mask_out, NUM_LANES);

endmodule
`endif

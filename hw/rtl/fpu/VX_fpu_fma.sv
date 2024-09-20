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

module VX_fpu_fma import VX_fpu_pkg::*; #(
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

    input wire [`INST_FRM_BITS-1:0] frm,

    input wire  is_madd,
    input wire  is_sub,
    input wire  is_neg,

    input wire [NUM_LANES-1:0][31:0]  dataa,
    input wire [NUM_LANES-1:0][31:0]  datab,
    input wire [NUM_LANES-1:0][31:0]  datac,
    output wire [NUM_LANES-1:0][31:0] result,

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAG_WIDTH-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    `UNUSED_VAR (frm)

    wire [NUM_LANES-1:0][3*32-1:0] data_in;
    wire [NUM_LANES-1:0] mask_out;
    wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+32)-1:0] data_out;
    wire [NUM_LANES-1:0][`FP_FLAGS_BITS-1:0] fflags_out;

    wire pe_enable;
    wire [NUM_PES-1:0][3*32-1:0] pe_data_in;
    wire [NUM_PES-1:0][(`FP_FLAGS_BITS+32)-1:0] pe_data_out;

    reg [NUM_LANES-1:0][31:0] a, b, c;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            if (is_madd) begin
                // MADD / MSUB / NMADD / NMSUB
                a[i] = is_neg ? {~dataa[i][31], dataa[i][30:0]} : dataa[i];
                b[i] = datab[i];
                c[i] = (is_neg ^ is_sub) ? {~datac[i][31], datac[i][30:0]} : datac[i];
            end else begin
                if (is_neg) begin
                    // MUL
                    a[i] = dataa[i];
                    b[i] = datab[i];
                    c[i] = '0;
                end else begin
                    // ADD / SUB
                    a[i] = 32'h3f800000; // 1.0f
                    b[i] = dataa[i];
                    c[i] = is_sub ? {~datab[i][31], datab[i][30:0]} : datab[i];
                end
            end
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign data_in[i][0  +: 32] = a[i];
        assign data_in[i][32 +: 32] = b[i];
        assign data_in[i][64 +: 32] = c[i];
    end

    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (`LATENCY_FMA),
        .DATA_IN_WIDTH(3*32),
        .DATA_OUT_WIDTH(`FP_FLAGS_BITS + 32),
        .TAG_WIDTH  (NUM_LANES + TAG_WIDTH),
        .PE_REG     ((NUM_LANES != NUM_PES) ? 1 : 0), // must be registered for DSPs
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

    fflags_t [NUM_LANES-1:0] per_lane_fflags;

`ifdef QUARTUS

    for (genvar i = 0; i < NUM_PES; ++i) begin
        acl_fmadd fmadd (
            .clk (clk),
            .areset (1'b0),
            .en (pe_enable),
            .a  (pe_data_in[i][0 +: 32]),
            .b  (pe_data_in[i][32 +: 32]),
            .c  (pe_data_in[i][64 +: 32]),
            .q  (pe_data_out[i][0 +: 32])
        );
        assign pe_data_out[i][32 +: `FP_FLAGS_BITS] = 'x;
    end

    assign has_fflags = 0;
    assign per_lane_fflags = 'x;

`elsif VIVADO

    for (genvar i = 0; i < NUM_PES; ++i) begin
        wire [2:0] tuser;

        xil_fma fma (
            .aclk                (clk),
            .aclken              (pe_enable),
            .s_axis_a_tvalid     (1'b1),
            .s_axis_a_tdata      (pe_data_in[i][0 +: 32]),
            .s_axis_b_tvalid     (1'b1),
            .s_axis_b_tdata      (pe_data_in[i][32 +: 32]),
            .s_axis_c_tvalid     (1'b1),
            .s_axis_c_tdata      (pe_data_in[i][64 +: 32]),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata (pe_data_out[i][0 +: 32]),
            .m_axis_result_tuser (tuser)
        );
                                                      // NV, DZ, OF, UF, NX
        assign pe_data_out[i][32 +: `FP_FLAGS_BITS] = {tuser[2], 1'b0, tuser[1], tuser[0], 1'b0};
    end

    assign has_fflags = 1;
    assign per_lane_fflags = fflags_out;

`else

    for (genvar i = 0; i < NUM_PES; ++i) begin
        reg [63:0] r;
        `UNUSED_VAR (r)
        fflags_t f;

        always @(*) begin
            dpi_fmadd (
                pe_enable,
                int'(0),
                {32'hffffffff, pe_data_in[i][0 +: 32]},
                {32'hffffffff, pe_data_in[i][32 +: 32]},
                {32'hffffffff, pe_data_in[i][64 +: 32]},
                frm,
                r,
                f
            );
        end

        VX_shift_register #(
            .DATAW  (32 + $bits(fflags_t)),
            .DEPTH  (`LATENCY_FMA)
        ) shift_req_dpi (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (pe_enable),
            .data_in  ({f, r[31:0]}),
            .data_out (pe_data_out[i])
        );
    end

    assign has_fflags = 1;
    assign per_lane_fflags = fflags_out;

`endif

`FPU_MERGE_FFLAGS(fflags, per_lane_fflags, mask_out, NUM_LANES);

endmodule

`endif

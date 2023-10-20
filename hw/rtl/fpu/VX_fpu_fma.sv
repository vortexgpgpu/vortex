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
    parameter TAGW = 1
) (
    input wire clk,
    input wire reset, 

    output wire ready_in,
    input wire  valid_in,

    input wire [NUM_LANES-1:0] lane_mask,

    input wire [TAGW-1:0] tag_in,
    
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

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    `UNUSED_VAR (frm)

    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    fflags_t [NUM_LANES-1:0] per_lane_fflags;
    wire [NUM_LANES-1:0] lane_mask_out;

    VX_shift_register #(
        .DATAW  (1 + NUM_LANES + TAGW),
        .DEPTH  (`LATENCY_FMA),
        .RESETW (1)
    ) shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in, lane_mask, tag_in}),
        .data_out ({valid_out, lane_mask_out, tag_out})
    );

    assign ready_in = enable;

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

`ifdef QUARTUS
    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        acl_fmadd fmadd (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (a[i]),
            .b      (b[i]),
            .c      (c[i]),
            .q      (result[i])
        );
    end
    
    assign has_fflags = 0;
    assign per_lane_fflags = 'x;

`elsif VIVADO

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [2:0] tuser;
        
        xil_fma fma (
            .aclk                (clk),
            .aclken              (enable),
            .s_axis_a_tvalid     (1'b1),
            .s_axis_a_tdata      (a[i]),
            .s_axis_b_tvalid     (1'b1),
            .s_axis_b_tdata      (b[i]),
            .s_axis_c_tvalid     (1'b1),
            .s_axis_c_tdata      (c[i]),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata (result[i]),
            .m_axis_result_tuser (tuser)
        );
                                    // NV, DZ, OF, UF, NX
        assign per_lane_fflags[i] = {tuser[2], 1'b0, tuser[1], tuser[0], 1'b0};
    end

    assign has_fflags = 1;

`else

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        reg [63:0] r;
        `UNUSED_VAR (r)

        fflags_t f;

        always @(*) begin        
            dpi_fmadd (enable && valid_in, int'(0), {32'hffffffff, a[i]}, {32'hffffffff, b[i]}, {32'hffffffff, c[i]}, frm, r, f);
        end

        VX_shift_register #(
            .DATAW  (32 + $bits(fflags_t)),
            .DEPTH  (`LATENCY_FMA)
        ) shift_req_dpi (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (enable),
            .data_in  ({r[31:0], f}),
            .data_out ({result[i], per_lane_fflags[i]})
        );
    end

    assign has_fflags = 1;

`endif

`FPU_MERGE_FFLAGS(fflags, per_lane_fflags, lane_mask_out, NUM_LANES);

endmodule
`endif

//!/bin/bash

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

`include "VX_rop_define.vh"

module VX_rop_blend import VX_rop_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 4,
    parameter TAG_WIDTH = 1
) (
    input wire  clk,
    input wire  reset,

    // DCRs
    input rop_dcrs_t dcrs,

    // Handshake
    input wire                  valid_in,
    input wire [TAG_WIDTH-1:0]  tag_in,
    output wire                 ready_in,    
    
    output wire                 valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    input wire                  ready_out,

    // Input values 
    input rgba_t [NUM_LANES-1:0] src_color,
    input rgba_t [NUM_LANES-1:0] dst_color,
    
    // Output values
    output rgba_t [NUM_LANES-1:0] color_out
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam LATENCY = 3;

    `UNUSED_VAR (dcrs)

    wire stall = ~ready_out && valid_out;
    
    assign ready_in = ~stall;
    
    rgba_t [NUM_LANES-1:0]  src_factor;
    rgba_t [NUM_LANES-1:0]  dst_factor;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : blend_func_inst
        VX_rop_blend_func #(
        ) rop_blend_func_src (
            .func_rgb   (dcrs.blend_src_rgb),
            .func_a     (dcrs.blend_src_a),
            .src_color  (src_color[i]),
            .dst_color  (dst_color[i]),
            .cst_color  (dcrs.blend_const),
            .factor_out (src_factor[i])
        );

        VX_rop_blend_func #(
        ) rop_blend_func_dst (
            .func_rgb   (dcrs.blend_dst_rgb),
            .func_a     (dcrs.blend_dst_a),
            .src_color  (src_color[i]),
            .dst_color  (dst_color[i]),
            .cst_color  (dcrs.blend_const),
            .factor_out (dst_factor[i])
        );
    end

    wire                 valid_s1, valid_s2;
    wire [TAG_WIDTH-1:0] tag_s1, tag_s2;

    rgba_t [NUM_LANES-1:0] src_color_s1;
    rgba_t [NUM_LANES-1:0] dst_color_s1;
    rgba_t [NUM_LANES-1:0] src_factor_s1;
    rgba_t [NUM_LANES-1:0] dst_factor_s1;

    VX_pipe_register #(
        .DATAW  (1 + TAG_WIDTH + 32 * 4 * NUM_LANES),
        .DEPTH  (2),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in, tag_in, src_color,    dst_color,    src_factor,    dst_factor}),
        .data_out ({valid_s1, tag_s1, src_color_s1, dst_color_s1, src_factor_s1, dst_factor_s1})
    );

    rgba_t [NUM_LANES-1:0] mult_add_color_s2;
    rgba_t [NUM_LANES-1:0] min_color_s2;
    rgba_t [NUM_LANES-1:0] max_color_s2;
    rgba_t [NUM_LANES-1:0] logic_op_color_s2;
    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_rop_blend_multadd #(
            .LATENCY (LATENCY)
        ) rop_blend_multadd (
            .clk        (clk),
            .reset      (reset),
            .enable     (~stall),            
            .mode_rgb   (dcrs.blend_mode_rgb),
            .mode_a     (dcrs.blend_mode_a),
            .src_color  (src_color_s1[i]),
            .dst_color  (dst_color_s1[i]),
            .src_factor (src_factor_s1[i]),
            .dst_factor (dst_factor_s1[i]),
            .color_out  (mult_add_color_s2[i])
        );

        VX_rop_blend_minmax #(
            .LATENCY (LATENCY)
        ) rop_blend_minmax (
            .clk        (clk),
            .reset      (reset),
            .enable     (~stall),
            .src_color  (src_color_s1[i]),
            .dst_color  (dst_color_s1[i]),
            .min_out    (min_color_s2[i]),
            .max_out    (max_color_s2[i])
        );

        VX_rop_logic_op #(
            .LATENCY (LATENCY)
        ) rop_logic_op (
            .clk        (clk),
            .reset      (reset),
            .enable     (~stall),
            .op         (dcrs.logic_op),
            .src_color  (src_color_s1[i]),
            .dst_color  (dst_color_s1[i]),
            .color_out  (logic_op_color_s2[i])
        );
    end

    VX_shift_register #(
        .DATAW  (1 + TAG_WIDTH),
        .DEPTH  (LATENCY)
    ) shift_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_s1, tag_s1}),
        .data_out ({valid_s2, tag_s2})
    );

    rgba_t [NUM_LANES-1:0] color_out_s2;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            // RGB Component
            case (dcrs.blend_mode_rgb)
                `VX_ROP_BLEND_MODE_ADD, 
                `VX_ROP_BLEND_MODE_SUB, 
                `VX_ROP_BLEND_MODE_REV_SUB: begin
                    color_out_s2[i].r = mult_add_color_s2[i].r;
                    color_out_s2[i].g = mult_add_color_s2[i].g;
                    color_out_s2[i].b = mult_add_color_s2[i].b;
                    end
                `VX_ROP_BLEND_MODE_MIN: begin
                    color_out_s2[i].r = min_color_s2[i].r;
                    color_out_s2[i].g = min_color_s2[i].g;
                    color_out_s2[i].b = min_color_s2[i].b;
                    end
                `VX_ROP_BLEND_MODE_MAX: begin
                    color_out_s2[i].r = max_color_s2[i].r;
                    color_out_s2[i].g = max_color_s2[i].g;
                    color_out_s2[i].b = max_color_s2[i].b;
                    end
                `VX_ROP_BLEND_MODE_LOGICOP: begin
                    color_out_s2[i].r = logic_op_color_s2[i].r;
                    color_out_s2[i].g = logic_op_color_s2[i].g;
                    color_out_s2[i].b = logic_op_color_s2[i].b;
                    end
                default: begin
                    color_out_s2[i].r = 'x;
                    color_out_s2[i].g = 'x;
                    color_out_s2[i].b = 'x;
                    end
            endcase
            // Alpha Component
            case (dcrs.blend_mode_a)
                `VX_ROP_BLEND_MODE_ADD, 
                `VX_ROP_BLEND_MODE_SUB, 
                `VX_ROP_BLEND_MODE_REV_SUB: begin
                    color_out_s2[i].a = mult_add_color_s2[i].a;
                    end
                `VX_ROP_BLEND_MODE_MIN: begin
                    color_out_s2[i].a = min_color_s2[i].a;
                    end
                `VX_ROP_BLEND_MODE_MAX: begin
                    color_out_s2[i].a = max_color_s2[i].a;
                    end
                `VX_ROP_BLEND_MODE_LOGICOP: begin
                    color_out_s2[i].a = logic_op_color_s2[i].a;
                    end
                default: begin
                    color_out_s2[i].a = 'x;
                    end
            endcase
        end
    end

    VX_pipe_register #(
        .DATAW  (1 + TAG_WIDTH + 32 * NUM_LANES),
        .RESETW (1)
    ) pipe_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_s2,  tag_s2,  color_out_s2}),
        .data_out ({valid_out, tag_out, color_out})
    );

endmodule

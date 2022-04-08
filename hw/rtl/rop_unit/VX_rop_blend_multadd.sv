`include "VX_rop_define.vh"

import VX_rop_types::*;

module VX_rop_blend_multadd #(
    //--
) (
    // Mode used to determine which values to multiply and whether to use addition or subtraction.
    input wire [`ROP_BLEND_MODE_BITS-1:0] mode_rgb,
    input wire [`ROP_BLEND_MODE_BITS-1:0] mode_a,

    input rgba_t src_color,
    input rgba_t dst_color,

    input rgba_t src_factor,
    input rgba_t dst_factor,

    output rgba_t color_out
);

    wire [15:0] product_src_r = (src_color.r * src_factor.r + 16'hff);
    wire [15:0] product_src_g = (src_color.g * src_factor.g + 16'hff);
    wire [15:0] product_src_b = (src_color.b * src_factor.b + 16'hff);
    wire [15:0] product_src_a = (src_color.a * src_factor.a + 16'hff);

    wire [15:0] product_dst_r = (dst_color.r * dst_factor.r + 16'hff);
    wire [15:0] product_dst_g = (dst_color.g * dst_factor.g + 16'hff);
    wire [15:0] product_dst_b = (dst_color.b * dst_factor.b + 16'hff);
    wire [15:0] product_dst_a = (dst_color.a * dst_factor.a + 16'hff);

    reg [15:0] sum_r, sum_g, sum_b, sum_a;

    always @(*) begin
        // RGB blending
        case(mode_rgb)
            `ROP_BLEND_MODE_ADD: begin
                sum_r = product_src_r + product_dst_r;
                sum_g = product_src_g + product_dst_g;
                sum_b = product_src_b + product_dst_b;
            end
            `ROP_BLEND_MODE_SUB: begin
                sum_r = product_src_r - product_dst_r;
                sum_g = product_src_g - product_dst_g;
                sum_b = product_src_b - product_dst_b; 
            end
            `ROP_BLEND_MODE_REV_SUB: begin
                sum_r = product_dst_r - product_src_r;
                sum_g = product_dst_g - product_src_g;
                sum_b = product_dst_b - product_src_b;
            end
            default: begin
                sum_r = 'x;
                sum_g = 'x;
                sum_b = 'x;
            end
        endcase
        // Alpha blending
        case(mode_a)
            `ROP_BLEND_MODE_ADD: begin
                sum_a = product_src_a + product_dst_a;
            end
            `ROP_BLEND_MODE_SUB: begin
                sum_a = product_src_a - product_dst_a;
            end
            `ROP_BLEND_MODE_REV_SUB: begin
                sum_a = product_dst_a - product_src_a;
            end
            default: begin
                sum_a = 'x;
            end
        endcase
    end

    assign color_out = {sum_a[15:8], sum_r[15:8], sum_g[15:8], sum_b[15:8]};

    `UNUSED_VAR (sum_r)
    `UNUSED_VAR (sum_g)
    `UNUSED_VAR (sum_b)
    `UNUSED_VAR (sum_a)

endmodule

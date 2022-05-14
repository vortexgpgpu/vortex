`include "VX_rop_define.vh"

`define MULT8(clk, en, dst, src1, src2) \
    VX_multiplier #(              \
        .WIDTHA  (8),             \
        .WIDTHB  (8),             \
        .WIDTHP  (16),            \
        .LATENCY (`LATENCY_IMUL)  \
    ) __``dst (                   \
        .clk    (clk),            \
        .enable (en),             \
        .dataa  (src1),           \
        .datab  (src2),           \
        .result (dst)             \
    )

module VX_rop_blend_multadd #(
    parameter LATENCY = (`LATENCY_IMUL + 1)
) (
    input clk,
    input reset,
    input enable,

    input wire [`ROP_BLEND_MODE_BITS-1:0] mode_rgb,
    input wire [`ROP_BLEND_MODE_BITS-1:0] mode_a,

    input rgba_t src_color,
    input rgba_t dst_color,

    input rgba_t src_factor,
    input rgba_t dst_factor,

    output rgba_t color_out
);
    `STATIC_ASSERT((LATENCY >= `LATENCY_IMUL), ("invalid parameter"))

    localparam LATENCY_REM = LATENCY - `LATENCY_IMUL;

    wire [15:0] prod_src_r, prod_src_g, prod_src_b, prod_src_a;
    wire [15:0] prod_dst_r, prod_dst_g, prod_dst_b, prod_dst_a;

    `MULT8(clk, enable, prod_src_r, src_color.r, src_factor.r);
    `MULT8(clk, enable, prod_src_g, src_color.g, src_factor.g);
    `MULT8(clk, enable, prod_src_b, src_color.b, src_factor.b);
    `MULT8(clk, enable, prod_src_a, src_color.a, src_factor.a);

    `MULT8(clk, enable, prod_dst_r, dst_color.r, dst_factor.r);
    `MULT8(clk, enable, prod_dst_g, dst_color.g, dst_factor.g);
    `MULT8(clk, enable, prod_dst_b, dst_color.b, dst_factor.b);
    `MULT8(clk, enable, prod_dst_a, dst_color.a, dst_factor.a);

    reg [15:0] sum_r, sum_g, sum_b, sum_a;

    always @(*) begin
        // RGB blending
        case (mode_rgb)
            `ROP_BLEND_MODE_ADD: begin
                sum_r = prod_src_r + prod_dst_r + 16'hff;
                sum_g = prod_src_g + prod_dst_g + 16'hff;
                sum_b = prod_src_b + prod_dst_b + 16'hff;
            end
            `ROP_BLEND_MODE_SUB: begin
                sum_r = prod_src_r - prod_dst_r + 16'hff;
                sum_g = prod_src_g - prod_dst_g + 16'hff;
                sum_b = prod_src_b - prod_dst_b + 16'hff; 
            end
            `ROP_BLEND_MODE_REV_SUB: begin
                sum_r = prod_dst_r - prod_src_r + 16'hff;
                sum_g = prod_dst_g - prod_src_g + 16'hff;
                sum_b = prod_dst_b - prod_src_b + 16'hff;
            end
            default: begin
                sum_r = 'x;
                sum_g = 'x;
                sum_b = 'x;
            end
        endcase
        // Alpha blending
        case (mode_a)
            `ROP_BLEND_MODE_ADD: begin
                sum_a = prod_src_a + prod_dst_a + 16'hff;
            end
            `ROP_BLEND_MODE_SUB: begin
                sum_a = prod_src_a - prod_dst_a + 16'hff;
            end
            `ROP_BLEND_MODE_REV_SUB: begin
                sum_a = prod_dst_a - prod_src_a + 16'hff;
            end
            default: begin
                sum_a = 'x;
            end
        endcase
    end

    `UNUSED_VAR (sum_r)
    `UNUSED_VAR (sum_g)
    `UNUSED_VAR (sum_b)
    `UNUSED_VAR (sum_a)

    VX_shift_register #(
        .DATAW  (32),
        .DEPTH  (LATENCY_REM)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({sum_a[15:8], sum_r[15:8], sum_g[15:8], sum_b[15:8]}),
        .data_out ({color_out.a, color_out.r, color_out.g, color_out.b})
    );

endmodule

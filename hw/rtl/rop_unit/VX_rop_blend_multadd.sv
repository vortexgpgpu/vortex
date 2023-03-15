`include "VX_rop_define.vh"

`define MULT8(clk, en, dst, src1, src2) \
    VX_multiplier #(              \
        .A_WIDTH (8),             \
        .B_WIDTH (8),             \
        .R_WIDTH (16),            \
        .LATENCY (`LATENCY_IMUL)  \
    ) __``dst (                   \
        .clk    (clk),            \
        .enable (en),             \
        .dataa  (src1),           \
        .datab  (src2),           \
        .result (dst)             \
    )

module VX_rop_blend_multadd #(
    parameter LATENCY = 1
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input wire [`ROP_BLEND_MODE_BITS-1:0] mode_rgb,
    input wire [`ROP_BLEND_MODE_BITS-1:0] mode_a,

    input rgba_t src_color,
    input rgba_t dst_color,

    input rgba_t src_factor,
    input rgba_t dst_factor,

    output rgba_t color_out
);

    `STATIC_ASSERT((LATENCY > `LATENCY_IMUL), ("invalid parameter"))
    `UNUSED_VAR (reset)

    localparam LATENCY_REM = LATENCY - `LATENCY_IMUL;

    wire [15:0] prod_src_r, prod_src_g, prod_src_b, prod_src_a;
    wire [15:0] prod_dst_r, prod_dst_g, prod_dst_b, prod_dst_a;

    // src_color x src_factor
    `MULT8(clk, enable, prod_src_r, src_color.r, src_factor.r);
    `MULT8(clk, enable, prod_src_g, src_color.g, src_factor.g);
    `MULT8(clk, enable, prod_src_b, src_color.b, src_factor.b);
    `MULT8(clk, enable, prod_src_a, src_color.a, src_factor.a);

    // dst_color x dst_factor
    `MULT8(clk, enable, prod_dst_r, dst_color.r, dst_factor.r);
    `MULT8(clk, enable, prod_dst_g, dst_color.g, dst_factor.g);
    `MULT8(clk, enable, prod_dst_b, dst_color.b, dst_factor.b);
    `MULT8(clk, enable, prod_dst_a, dst_color.a, dst_factor.a);

    reg [16:0] sum_r, sum_g, sum_b, sum_a;

    // apply blend mode
    always @(*) begin
        case (mode_rgb)
            `ROP_BLEND_MODE_ADD: begin
                sum_r = prod_src_r + prod_dst_r;
                sum_g = prod_src_g + prod_dst_g;
                sum_b = prod_src_b + prod_dst_b;
            end
            `ROP_BLEND_MODE_SUB: begin
                sum_r = prod_src_r - prod_dst_r;
                sum_g = prod_src_g - prod_dst_g;
                sum_b = prod_src_b - prod_dst_b; 
            end
            `ROP_BLEND_MODE_REV_SUB: begin
                sum_r = prod_dst_r - prod_src_r;
                sum_g = prod_dst_g - prod_src_g;
                sum_b = prod_dst_b - prod_src_b;
            end
            default: begin
                sum_r = 'x;
                sum_g = 'x;
                sum_b = 'x;
            end
        endcase
        case (mode_a)
            `ROP_BLEND_MODE_ADD: begin
                sum_a = prod_src_a + prod_dst_a;
            end
            `ROP_BLEND_MODE_SUB: begin
                sum_a = prod_src_a - prod_dst_a;
            end
            `ROP_BLEND_MODE_REV_SUB: begin
                sum_a = prod_dst_a - prod_src_a;
            end
            default: begin
                sum_a = 'x;
            end
        endcase
    end

    reg [15:0] clamp_r, clamp_g, clamp_b, clamp_a;

    // clamp to (0, 255 * 255)
    always @(*) begin
        case (mode_rgb)
            `ROP_BLEND_MODE_ADD: begin
                clamp_r = (sum_r > 17'hFE01) ? 16'hFE01 : sum_r[15:0];
                clamp_g = (sum_g > 17'hFE01) ? 16'hFE01 : sum_g[15:0];
                clamp_b = (sum_b > 17'hFE01) ? 16'hFE01 : sum_b[15:0];
            end
            `ROP_BLEND_MODE_SUB,
            `ROP_BLEND_MODE_REV_SUB: begin
                clamp_r = sum_r[16] ? 16'h0 : sum_r[15:0];
                clamp_g = sum_g[16] ? 16'h0 : sum_g[15:0];
                clamp_b = sum_b[16] ? 16'h0 : sum_b[15:0];
            end
            default: begin
                clamp_r = 'x;
                clamp_g = 'x;
                clamp_b = 'x;
            end
        endcase
        case (mode_a)
            `ROP_BLEND_MODE_ADD: begin
                clamp_a = (sum_a > 17'hFE01) ? 16'hFE01 : sum_a[15:0];
            end
            `ROP_BLEND_MODE_SUB,
            `ROP_BLEND_MODE_REV_SUB: begin
                clamp_a = sum_a[16] ? 16'h0 : sum_a[15:0];
            end
            default: begin
                clamp_a = 'x;
            end
        endcase
    end

    rgba_t result;

    // divide by 255
    assign result.r = 8'((clamp_r + (clamp_r >> 8)) >> 8);
    assign result.g = 8'((clamp_g + (clamp_g >> 8)) >> 8);
    assign result.b = 8'((clamp_b + (clamp_b >> 8)) >> 8);
    assign result.a = 8'((clamp_a + (clamp_a >> 8)) >> 8);

    VX_shift_register #(
        .DATAW  (32),
        .DEPTH  (LATENCY_REM)
    ) shift_reg (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (enable),
        .data_in  (result),
        .data_out (color_out)
    );

endmodule

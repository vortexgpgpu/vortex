`include "VX_rop_define.vh"

module VX_rop_blend_multadd #(
    parameter LATENCY = 1
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input wire [`VX_ROP_BLEND_MODE_BITS-1:0] mode_rgb,
    input wire [`VX_ROP_BLEND_MODE_BITS-1:0] mode_a,

    input rgba_t src_color,
    input rgba_t dst_color,

    input rgba_t src_factor,
    input rgba_t dst_factor,

    output rgba_t color_out
);

    `STATIC_ASSERT((LATENCY == 3), ("invalid parameter"))
    `UNUSED_VAR (reset)

    // multiply-add

    reg [15:0] prod_src_r, prod_src_g, prod_src_b, prod_src_a;
    reg [15:0] prod_dst_r, prod_dst_g, prod_dst_b, prod_dst_a;
    reg [16:0] sum_r, sum_g, sum_b, sum_a;    

    always @(posedge clk) begin    
        if (enable) begin
            prod_src_r <= src_color.r * src_factor.r;
            prod_src_g <= src_color.g * src_factor.g;
            prod_src_b <= src_color.b * src_factor.b;
            prod_src_a <= src_color.a * src_factor.a;

            prod_dst_r <= dst_color.r * dst_factor.r;
            prod_dst_g <= dst_color.g * dst_factor.g;
            prod_dst_b <= dst_color.b * dst_factor.b;
            prod_dst_a <= dst_color.a * dst_factor.a;

            case (mode_rgb)
                `VX_ROP_BLEND_MODE_ADD: begin
                    sum_r <= prod_src_r + prod_dst_r + 16'h80;
                    sum_g <= prod_src_g + prod_dst_g + 16'h80;
                    sum_b <= prod_src_b + prod_dst_b + 16'h80;
                end
                `VX_ROP_BLEND_MODE_SUB: begin
                    sum_r <= prod_src_r - prod_dst_r + 16'h80;
                    sum_g <= prod_src_g - prod_dst_g + 16'h80;
                    sum_b <= prod_src_b - prod_dst_b + 16'h80; 
                end
                `VX_ROP_BLEND_MODE_REV_SUB: begin
                    sum_r <= prod_dst_r - prod_src_r + 16'h80;
                    sum_g <= prod_dst_g - prod_src_g + 16'h80;
                    sum_b <= prod_dst_b - prod_src_b + 16'h80;
                end
            endcase
            case (mode_a)
                `VX_ROP_BLEND_MODE_ADD: begin
                    sum_a <= prod_src_a + prod_dst_a + 16'h80;
                end
                `VX_ROP_BLEND_MODE_SUB: begin
                    sum_a <= prod_src_a - prod_dst_a + 16'h80;
                end
                `VX_ROP_BLEND_MODE_REV_SUB: begin
                    sum_a <= prod_dst_a - prod_src_a + 16'h80;
                end
            endcase
        end
    end

    // clamp to (0, 255 * 256)

    reg [15:0] clamp_r, clamp_g, clamp_b, clamp_a;

    always @(*) begin
        case (mode_rgb)
            `VX_ROP_BLEND_MODE_ADD: begin
                clamp_r = (sum_r > 17'hFF00) ? 16'hFF00 : sum_r[15:0];
                clamp_g = (sum_g > 17'hFF00) ? 16'hFF00 : sum_g[15:0];
                clamp_b = (sum_b > 17'hFF00) ? 16'hFF00 : sum_b[15:0];
            end
            `VX_ROP_BLEND_MODE_SUB,
            `VX_ROP_BLEND_MODE_REV_SUB: begin
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
            `VX_ROP_BLEND_MODE_ADD: begin
                clamp_a = (sum_a > 17'hFF00) ? 16'hFF00 : sum_a[15:0];
            end
            `VX_ROP_BLEND_MODE_SUB,
            `VX_ROP_BLEND_MODE_REV_SUB: begin
                clamp_a = sum_a[16] ? 16'h0 : sum_a[15:0];
            end
            default: begin
                clamp_a = 'x;
            end
        endcase
    end

    // divide by 255

    rgba_t result;
    assign result.r = 8'((clamp_r + (clamp_r >> 8)) >> 8);
    assign result.g = 8'((clamp_g + (clamp_g >> 8)) >> 8);
    assign result.b = 8'((clamp_b + (clamp_b >> 8)) >> 8);
    assign result.a = 8'((clamp_a + (clamp_a >> 8)) >> 8);

    VX_pipe_register #(
        .DATAW (32)
    ) pipe_reg (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (enable),
        .data_in  (result),
        .data_out (color_out)
    );

endmodule

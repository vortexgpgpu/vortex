`include "VX_rop_define.vh"

module VX_rop_blend_minmax #(
    parameter LATENCY = 1
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input rgba_t src_color,
    input rgba_t dst_color,

    output rgba_t min_out,
    output rgba_t max_out
);

    `UNUSED_VAR (reset)

    rgba_t tmp_min;
    rgba_t tmp_max;

    always @(*) begin   
        if (src_color.r > dst_color.r) begin
            tmp_max.r = src_color.r;
            tmp_min.r = dst_color.r;
        end else begin
            tmp_max.r = dst_color.r;
            tmp_min.r = src_color.r;
        end

        if (src_color.g > dst_color.g) begin
            tmp_max.g = src_color.g;
            tmp_min.g = dst_color.g;
        end else begin
            tmp_max.g = dst_color.g;
            tmp_min.g = src_color.g;
        end

        if (src_color.b > dst_color.b) begin
            tmp_max.b = src_color.b;
            tmp_min.b = dst_color.b;
        end else begin
            tmp_max.b = dst_color.b;
            tmp_min.b = src_color.b;
        end

        if (src_color.a > dst_color.a) begin
            tmp_max.a = src_color.a;
            tmp_min.a = dst_color.a;
        end else begin
            tmp_max.a = dst_color.a;
            tmp_min.a = src_color.a;
        end
    end

    VX_shift_register #(
        .DATAW (32 + 32),
        .DEPTH (LATENCY)
    ) shift_reg (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (enable),
        .data_in  ({tmp_max, tmp_min}),
        .data_out ({max_out, min_out})
    );

endmodule

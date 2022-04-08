`include "VX_rop_define.vh"

import VX_rop_types::*;

module VX_rop_blend_minmax #(
    //--
) (
    input rgba_t src_color,
    input rgba_t dst_color,

    output rgba_t min_out,
    output rgba_t max_out
);

    always @(*) begin
        // Compare Red Components
        if (src_color.r > dst_color.r) begin
            max_out.r = src_color.r;
            min_out.r = dst_color.r;
        end
        else begin
            max_out.r = dst_color.r;
            min_out.r = src_color.r;
        end
        // Compare Green Components
        if (src_color.g > dst_color.g) begin
            max_out.g = src_color.g;
            min_out.g = dst_color.g;
        end
        else begin
            max_out.g = dst_color.g;
            min_out.g = src_color.g;
        end
        // Compare Blue Components
        if (src_color.b > dst_color.b) begin
            max_out.b = src_color.b;
            min_out.b = dst_color.b;
        end
        else begin
            max_out.b = dst_color.b;
            min_out.b = src_color.b;
        end
        // Compare Alpha Components
        if (src_color.a > dst_color.a) begin
            max_out.a = src_color.a;
            min_out.a = dst_color.a;
        end
        else begin
            max_out.a = dst_color.a;
            min_out.a = src_color.a;
        end
    end


endmodule

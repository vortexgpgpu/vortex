`include "VX_rop_define.vh"

module VX_blend_func #(
    parameter INDEX = 0
) (
    input wire [`ROP_BLEND_FUNC_BITS-1:0] func, 
    input wire [3:0][7:0]  src, 
    input wire [3:0][7:0]  dst, 
    input wire [3:0][7:0]  cst,
    output reg [7:0]       out
);

    wire [7:0] one_minus_dst_a = 8'hFF - dst[3];

    always @(*) begin
        case (func)
            `ROP_BLEND_FUNC_ZERO:                 out = 8'h0;
            `ROP_BLEND_FUNC_ONE:                  out = 8'hFF;
            `ROP_BLEND_FUNC_SRC_RGB:              out = src[INDEX];
            `ROP_BLEND_FUNC_ONE_MINUS_SRC_RGB:    out = 8'hFF - src[INDEX];
            `ROP_BLEND_FUNC_SRC_A:                out = src[3];
            `ROP_BLEND_FUNC_ONE_MINUS_SRC_A:      out = 8'hFF - src[3];
            `ROP_BLEND_FUNC_DST_RGB:              out = dst[INDEX];
            `ROP_BLEND_FUNC_ONE_MINUS_DST_RGB:    out = 8'hFF - dst[INDEX];
            `ROP_BLEND_FUNC_DST_A:                out = dst[3];
            `ROP_BLEND_FUNC_ONE_MINUS_DST_A:      out = one_minus_dst_a;
            `ROP_BLEND_FUNC_CONST_RGB:            out = cst[INDEX];
            `ROP_BLEND_FUNC_ONE_MINUS_CONST_RGB:  out = 8'hFF - cst[INDEX];
            `ROP_BLEND_FUNC_CONST_A:              out = cst[3];
            `ROP_BLEND_FUNC_ONE_MINUS_CONST_A:    out = 8'hFF - cst[3];
            `ROP_BLEND_FUNC_ALPHA_SAT: begin
                if (INDEX < 3) begin
                    out = (src[3] < one_minus_dst_a) ? src[3] : one_minus_dst_a;
                end else begin
                    out = 8'hFF;
                end
            end
            default:                              out = 8'hx;
        endcase
    end
        
endmodule

module VX_rop_blend_func #(
    //--
) (
    input wire [`ROP_BLEND_FUNC_BITS-1:0] func_rgb,
    input wire [`ROP_BLEND_FUNC_BITS-1:0] func_a,

    input rgba_t src_color,
    input rgba_t dst_color,
    input rgba_t cst_color,

    output rgba_t factor_out
);
    VX_blend_func #(0) blend_func_b (func_rgb, src_color, dst_color, cst_color, factor_out.b);
    VX_blend_func #(1) blend_func_g (func_rgb, src_color, dst_color, cst_color, factor_out.g);  
    VX_blend_func #(2) blend_func_r (func_rgb, src_color, dst_color, cst_color, factor_out.r);
    VX_blend_func #(3) blend_func_a (func_a,   src_color, dst_color, cst_color, factor_out.a);

endmodule

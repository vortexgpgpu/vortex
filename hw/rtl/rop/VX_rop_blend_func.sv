`include "VX_rop_define.vh"

module VX_blend_func #(
    parameter INDEX = 0
) (
    input wire [`VX_ROP_BLEND_FUNC_BITS-1:0] func, 
    input wire [3:0][7:0]  src, 
    input wire [3:0][7:0]  dst, 
    input wire [3:0][7:0]  cst,
    output wire [7:0]      result
);

    wire [7:0] one_minus_dst_a = 8'hFF - dst[3];

    reg [7:0] result_r;

    always @(*) begin
        case (func)
            `VX_ROP_BLEND_FUNC_ZERO:                 result_r = 8'h0;
            `VX_ROP_BLEND_FUNC_ONE:                  result_r = 8'hFF;
            `VX_ROP_BLEND_FUNC_SRC_RGB:              result_r = src[INDEX];
            `VX_ROP_BLEND_FUNC_ONE_MINUS_SRC_RGB:    result_r = 8'hFF - src[INDEX];
            `VX_ROP_BLEND_FUNC_SRC_A:                result_r = src[3];
            `VX_ROP_BLEND_FUNC_ONE_MINUS_SRC_A:      result_r = 8'hFF - src[3];
            `VX_ROP_BLEND_FUNC_DST_RGB:              result_r = dst[INDEX];
            `VX_ROP_BLEND_FUNC_ONE_MINUS_DST_RGB:    result_r = 8'hFF - dst[INDEX];
            `VX_ROP_BLEND_FUNC_DST_A:                result_r = dst[3];
            `VX_ROP_BLEND_FUNC_ONE_MINUS_DST_A:      result_r = one_minus_dst_a;
            `VX_ROP_BLEND_FUNC_CONST_RGB:            result_r = cst[INDEX];
            `VX_ROP_BLEND_FUNC_ONE_MINUS_CONST_RGB:  result_r = 8'hFF - cst[INDEX];
            `VX_ROP_BLEND_FUNC_CONST_A:              result_r = cst[3];
            `VX_ROP_BLEND_FUNC_ONE_MINUS_CONST_A:    result_r = 8'hFF - cst[3];
            `VX_ROP_BLEND_FUNC_ALPHA_SAT: begin
                if (INDEX < 3) begin
                    result_r = (src[3] < one_minus_dst_a) ? src[3] : one_minus_dst_a;
                end else begin
                    result_r = 8'hFF;
                end
            end
            default:                              result_r = 8'hx;
        endcase
    end

    assign result = result_r;
        
endmodule

module VX_rop_blend_func #(
    //--
) (
    input wire [`VX_ROP_BLEND_FUNC_BITS-1:0] func_rgb,
    input wire [`VX_ROP_BLEND_FUNC_BITS-1:0] func_a,

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

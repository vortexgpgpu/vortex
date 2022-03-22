`include "VX_rop_define.vh"

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
    
    function logic[7:0] blend_func(logic [`ROP_BLEND_FUNC_BITS-1:0] func, 
                                   int i,
                                   logic [3:0][7:0] src, 
                                   logic [3:0][7:0] dst, 
                                   logic [3:0][7:0] cst);

        logic [7:0] one_minus_dst_a = 8'hFF - dst[3];

        case (func)
            `ROP_BLEND_FUNC_ZERO:                 return 8'h0;
            `ROP_BLEND_FUNC_ONE:                  return 8'hFF;
            `ROP_BLEND_FUNC_SRC_RGB:              return src[i];
            `ROP_BLEND_FUNC_ONE_MINUS_SRC_RGB:    return 8'hFF - src[i];
            `ROP_BLEND_FUNC_SRC_A:                return src[3];
            `ROP_BLEND_FUNC_ONE_MINUS_SRC_A:      return 8'hFF - src[3];
            `ROP_BLEND_FUNC_DST_RGB:              return dst[i];
            `ROP_BLEND_FUNC_ONE_MINUS_DST_RGB:    return 8'hFF - dst[i];
            `ROP_BLEND_FUNC_DST_A:                return dst[3];
            `ROP_BLEND_FUNC_ONE_MINUS_DST_A:      return one_minus_dst_a;
            `ROP_BLEND_FUNC_CONST_RGB:            return cst[i];
            `ROP_BLEND_FUNC_ONE_MINUS_CONST_RGB:  return 8'hFF - cst[i];
            `ROP_BLEND_FUNC_CONST_A:              return cst[3];
            `ROP_BLEND_FUNC_ONE_MINUS_CONST_A:    return 8'hFF - cst[3];
            `ROP_BLEND_FUNC_ALPHA_SAT:
                if (i < 3) begin
                    return (src[3] < one_minus_dst_a) ? src[3] : one_minus_dst_a;
                end else begin
                    return 8'hFF;
                end
            default:                              return 8'x;
        endcase
    endfunction

    assign factor_out.b = blend_func(func_rgb, 0, src_color, dst_color, cst_color);
    assign factor_out.g = blend_func(func_rgb, 1, src_color, dst_color, cst_color);  
    assign factor_out.r = blend_func(func_rgb, 2, src_color, dst_color, cst_color);
    assign factor_out.a = blend_func(func_a,   3, src_color, dst_color, cst_color);

endmodule

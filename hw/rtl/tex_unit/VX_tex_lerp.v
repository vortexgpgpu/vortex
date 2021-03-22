`include "VX_tex_define.vh"

module VX_tex_lerp #(
) (
    input wire [`BLEND_FRAC_64-1:0]   blend, 
    input wire [1:0][63:0]            in_texels,

    output wire [63:0]                lerp_texel
);  

    wire [63:0] lerp_i1;
    wire [63:0] lerp_i2; // >> BLEND_FRAC_64 / >> 8

    assign lerp_i1 = (in_texels[0] - in_texels[1]) * blend;
    assign lerp_i2 = in_texels[1] + {8'h00,lerp_i1[63:56], 8'h00,lerp_i1[47:40], 8'h00,lerp_i1[31:24], 8'h00,lerp_i1[15:8]};
    assign lerp_texel = lerp_i2 & 64'h00ff00ff00ff00ff;

endmodule
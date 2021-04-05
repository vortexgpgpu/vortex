`include "VX_tex_define.vh"

module VX_tex_lerp #(
) (
    input wire [`BLEND_FRAC-1:0] blend, 
    input wire  [31:0] in1,
    input wire  [31:0] in2,
    output wire [31:0] out
);  
    for (genvar i = 0; i < 4; ++i) begin
        wire [8:0]  blend_m1 = `BLEND_ONE - blend;
        wire [16:0] sum = in1[i*8+:8] * blend_m1 + in2[i*8+:8] * blend;
        `UNUSED_VAR (sum)
        assign out[i*8+:8] = sum[15:8];
    end

endmodule
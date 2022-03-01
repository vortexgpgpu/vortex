`include "VX_tex_define.vh"

module VX_tex_lerp (
    input wire [3:0][7:0]  in1,
    input wire [3:0][7:0]  in2,
    input wire [7:0]       frac,
    output wire [3:0][7:0] out
);
    for (genvar i = 0; i < 4; ++i) begin
        wire [16:0] sum = in1[i] * 9'(9'd256 - frac) + in2[i] * frac;
        `UNUSED_VAR (sum)
        assign out[i] = sum[15:8];
    end

endmodule
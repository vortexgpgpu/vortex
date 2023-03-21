`include "VX_platform.vh"

`TRACING_OFF
module VX_tex_lerp (
    input wire [7:0]  in1,
    input wire [7:0]  in2,
    input wire [7:0]  frac,
    output wire [7:0] out
);
    wire [7:0] sub = (8'hff - frac);
    wire [16:0] tmp = in1 * sub + in2 * frac + 16'h80;
    assign out = 8'((tmp + (tmp >> 8)) >> 8);

endmodule
`TRACING_ON

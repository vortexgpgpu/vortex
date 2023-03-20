`include "VX_platform.vh"

`TRACING_OFF
module VX_lerp_fx #(
    parameter N = 8,
    parameter F = N
) (
    input wire [N-1:0]  in1,
    input wire [N-1:0]  in2,
    input wire [F-1:0]  frac,
    output wire [N-1:0] out
);
    wire [F-1:0] One  = {F{1'b1}};
    wire [F-1:0] Half = One >> 1;
    wire [F-1:0] sub  = One - frac;
    wire [N+F:0] tmp  = in1 * sub + in2 * frac + (N+F+1)'(Half);
    assign out = N'((tmp + (tmp >> F)) >> F);

endmodule
`TRACING_ON

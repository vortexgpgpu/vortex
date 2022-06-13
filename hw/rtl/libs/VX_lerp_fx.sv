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

    wire [F:0]     one  = {1'b1, {F{1'b0}}};
    wire [F+1:0]   sub  = one - (F+1)'(frac);
    wire [(N+F):0] prod = in1 * sub + in2 * frac;
    `UNUSED_VAR (prod)

    assign out = prod [F +: N];

endmodule
`TRACING_ON

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
    localparam N2  = 2 * N;
    localparam ONE = 1 << F;

    wire [F:0] sub = (F+1)'(ONE) - frac;
    wire [(N+F):0] sum = in1 * sub + in2 * frac;
    `UNUSED_VAR (sum)
    
    assign out = sum [F +: N];

endmodule
`TRACING_ON
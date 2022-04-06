`include "VX_platform.vh"

`TRACING_OFF
module VX_lerp_fx #(
    parameter N = 8
) (
    input wire [N-1:0]  in1,
    input wire [N-1:0]  in2,
    input wire [N-1:0]  frac,
    output wire [N-1:0] out
);
    localparam N2  = 2 * N;
    localparam ONE = 1 << N;

    wire [N2:0] sum = in1 * (N+1)'((N+1)'(ONE) - frac) + in2 * frac;
    `UNUSED_VAR (sum)
    
    assign out = sum[N2-1:N];

endmodule
`TRACING_ON
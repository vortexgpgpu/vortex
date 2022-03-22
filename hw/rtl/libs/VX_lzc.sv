`include "VX_platform.vh"

`TRACING_OFF
module VX_lzc #(
    parameter N    = 2,
    parameter MODE = 0, // 0 -> trailing zero, 1 -> leading zero
    parameter LOGN = $clog2(N)
) (
    input  wire [N-1:0]    in_i,
    output wire [LOGN-1:0] cnt_o,
    output wire            valid_o
);
    wire [N-1:0][LOGN-1:0] indices;

    for (genvar i = 0; i < N; ++i) begin
        assign indices[i] = MODE ? LOGN'(N-1-i) : LOGN'(i);
    end

    VX_find_first #(
        .N       (N),
        .DATAW   (LOGN),
        .REVERSE (MODE)
    ) find_first (        
        .data_i  (indices),
        .valid_i (in_i),
        .data_o  (cnt_o),
        .valid_o (valid_o)
    );
  
endmodule
`TRACING_ON

`include "VX_platform.vh"

`TRACING_OFF
module VX_lzc #(
    parameter N       = 2,
    parameter REVERSE = 0,  // 0 -> leading zero, 1 -> trailing zero,
    parameter LOGN    = `LOG2UP(N)
) (
    input  wire [N-1:0]    data_in,
    output wire [LOGN-1:0] data_out,
    output wire            valid_out
);
    if (N == 1) begin

        `UNUSED_PARAM (REVERSE)

        assign data_out  = 0;
        assign valid_out = data_in;

    end else begin

        wire [N-1:0][LOGN-1:0] indices;

        for (genvar i = 0; i < N; ++i) begin
            assign indices[i] = REVERSE ? LOGN'(i) : LOGN'(N-1-i);
        end

        VX_find_first #(
            .N       (N),
            .DATAW   (LOGN),
            .REVERSE (!REVERSE)
        ) find_first (        
            .data_in   (indices),
            .valid_in  (data_in),
            .data_out  (data_out),
            .valid_out (valid_out)
        );

    end
  
endmodule
`TRACING_ON

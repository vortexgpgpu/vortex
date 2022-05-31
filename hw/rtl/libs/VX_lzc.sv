`include "VX_platform.vh"

`TRACING_OFF
module VX_lzc #(
    parameter N       = 2,
    parameter REVERSE = 0,  // 0 -> leading zero, 1 -> trailing zero,
    localparam LOGN   = `LOG2UP(N)
) (
    input  wire [N-1:0]    in_i,
    output wire [LOGN-1:0] cnt_o,
    output wire            valid_o
);
    if (N == 1) begin

        `UNUSED_PARAM (REVERSE)

        assign cnt_o   = 0;
        assign valid_o = in_i;

    end else if (N == 2) begin

        assign cnt_o   = in_i[REVERSE];
        assign valid_o = (| in_i);

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
            .data_i  (indices),
            .valid_i (in_i),
            .data_o  (cnt_o),
            .valid_o (valid_o)
        );

    end
  
endmodule
`TRACING_ON

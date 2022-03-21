`include "VX_platform.vh"

`TRACING_OFF
module VX_find_first #(
    parameter N       = 1,
    parameter DATAW   = 1,
    parameter REVERSE = 0,
    parameter LOGN    = $clog2(N)
) (
    input  wire [N-1:0][DATAW-1:0] data_i,
    input  wire [N-1:0]            valid_i,    
    output wire [DATAW-1:0]        data_o,
    output wire                    valid_o
);
    localparam TL = (1 << LOGN) - 1;
    localparam TN = (1 << (LOGN+1)) - 1;

`IGNORE_WARNINGS_BEGIN
    wire [TN-1:0] s_n;
    wire [TN-1:0][DATAW-1:0] d_n;
`IGNORE_WARNINGS_END

    for (genvar i = 0; i < N; ++i) begin
        assign s_n[TL+i] = REVERSE ? valid_i[N-1-i] : valid_i[i];
        assign d_n[TL+i] = REVERSE ? data_i[N-1-i] : data_i[i];
    end
    
    for (genvar i = TL+N; i < TN; ++i) begin
        assign s_n[i] = 0;
        assign d_n[i] = 'x;
    end

    for (genvar j = 0; j < LOGN; ++j) begin
        for (genvar i = 0; i < (2**j); ++i) begin
            assign s_n[2**j-1+i] = s_n[2**(j+1)-1+i*2] | s_n[2**(j+1)-1+i*2+1];
            assign d_n[2**j-1+i] = s_n[2**(j+1)-1+i*2] ? d_n[2**(j+1)-1+i*2] : d_n[2**(j+1)-1+i*2+1];
        end
    end     
        
    assign valid_o = s_n[0];
    assign data_o  = d_n[0];
  
endmodule
`TRACING_ON

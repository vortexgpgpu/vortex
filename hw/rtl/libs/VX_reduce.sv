`include "VX_platform.vh"

`TRACING_OFF
module VX_reduce #(    
    parameter DATAW     = 1,
    parameter N         = 1,
    parameter string OP = "+"
) (
    input wire [N-1:0][DATAW-1:0] data_in,
    output wire [DATAW-1:0]       data_out
);
    wire [N-1:0][DATAW-1:0] tmp;

    assign tmp[0] = data_in[0];

    for (genvar i = 1; i < N; ++i) begin                    
        if (OP == "+") begin
            assign tmp[i] = tmp[i-1] + data_in[i];
        end if (OP == "^") begin
            assign tmp[i] = tmp[i-1] ^ data_in[i];
        end if (OP == "&") begin
            assign tmp[i] = tmp[i-1] & data_in[i];
        end else if (OP == "|") begin
            assign tmp[i] = tmp[i-1] | data_in[i];
        end else begin
            `ERROR(("invalid parameter"));        
        end
    end    

    assign data_out = tmp[N-1];
  
endmodule
`TRACING_ON
`include "VX_platform.vh"

module VX_lzc #(
    parameter DATAW  = 32,
    parameter LDATAW = `LOG2UP(DATAW)
) (
    input wire  [DATAW-1:0]  data_in,
    output wire [LDATAW-1:0] data_out,
    output wire              valid_out
); 

    reg [LDATAW-1:0] data_out_r;

    always @(*) begin
        data_out_r = 'x;
        for (integer i = DATAW-1; i >= 0; --i) begin
            if (data_in[i]) begin
                data_out_r = LDATAW'(DATAW-1-i);
                break;
            end
        end
    end

    assign data_out  = data_out_r;
    assign valid_out = (| data_in);
  
endmodule
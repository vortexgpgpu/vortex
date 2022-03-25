`include "VX_platform.vh"

module VX_reset_relay #(
    parameter N     = 1,
    parameter DEPTH = 1
) (
    input wire  clk,
    input wire  reset,
    output wire [N-1:0] reset_o
); 
    
    if (DEPTH > 1) begin
        `PRESERVE_REG `DISABLE_BRAM reg [N-1:0] reset_r [DEPTH-1:0];
        always @(posedge clk) begin
            for (integer i = DEPTH-1; i > 0; --i)
                reset_r[i] <= reset_r[i-1];
            reset_r[0] <= {N{reset}};
        end       
        assign reset_o = reset_r[DEPTH-1];
    end else if (DEPTH == 1) begin
        `PRESERVE_REG reg [N-1:0] reset_r;
        always @(posedge clk) begin
            reset_r <= {N{reset}};
        end
        assign reset_o = reset_r;
    end else begin
        `UNUSED_VAR (clk)
        assign reset_o = {N{reset}};
    end
  
endmodule

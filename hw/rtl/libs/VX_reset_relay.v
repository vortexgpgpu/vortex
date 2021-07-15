`include "VX_platform.vh"

module VX_reset_relay #(
    parameter ASYNC = 0
) (
    input wire  clk,
    input wire  reset,
    output wire reset_o
); 
    (* preserve *) reg reset_r;
    
    if (ASYNC) begin
        always @(posedge clk or posedge reset) begin
            reset_r <= reset;
        end
    end else begin
        always @(posedge clk) begin
            reset_r <= reset;
        end
    end

    assign reset_o = reset_r;
  
endmodule
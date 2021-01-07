`include "VX_platform.vh"

module VX_reset_relay #(
    parameter NUM_NODES = 1,
    parameter PASSTHRU  = 0
) (
    input wire  clk,
    input wire  reset,
    output wire [NUM_NODES-1:0] reset_out
); 
    
    if (PASSTHRU == 0) begin
        reg [NUM_NODES-1:0] reset_r;    
        always @(posedge clk) begin
            for (integer i = 0; i < NUM_NODES; ++i) begin
                reset_r[i] <= reset;
            end
        end       
        assign reset_out = reset_r;
    end else begin
        `UNUSED_VAR (clk)
        for (genvar i = 0; i < NUM_NODES; ++i) begin
            assign reset_out[i] = reset;
        end
    end
  
endmodule
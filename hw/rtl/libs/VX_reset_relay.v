`include "VX_platform.vh"

module VX_reset_relay #(
    parameter NUM_NODES = 1,
    parameter DEPTH     = 1,
    parameter ASYNC     = 0
) (
    input wire  clk,
    input wire  reset,
    output wire [NUM_NODES-1:0] reset_o
); 
    
    if (DEPTH > 1) begin
        `DISABLE_BRAM reg [NUM_NODES-1:0] reset_r [DEPTH-1:0];
        if (ASYNC) begin
            always @(posedge clk or posedge reset) begin
                for (integer i = DEPTH-1; i > 0; --i)
                    reset_r[i] <= reset_r[i-1];
                reset_r[0] <= {NUM_NODES{reset}};                
            end
        end else begin
            always @(posedge clk) begin
                for (integer i = DEPTH-1; i > 0; --i)
                    reset_r[i] <= reset_r[i-1];
                reset_r[0] <= {NUM_NODES{reset}};
            end
        end       
        assign reset_o = reset_r[DEPTH-1];
    end else if (DEPTH == 1) begin
        reg [NUM_NODES-1:0] reset_r;
        if (ASYNC) begin
            always @(posedge clk or posedge reset) begin
                reset_r <= {NUM_NODES{reset}};
            end
        end else begin
            always @(posedge clk) begin
                reset_r <= {NUM_NODES{reset}};
            end
        end
        assign reset_o = reset_r;
    end else begin
        `UNUSED_VAR (clk)
        assign reset_o = {NUM_NODES{reset}};
    end
  
endmodule
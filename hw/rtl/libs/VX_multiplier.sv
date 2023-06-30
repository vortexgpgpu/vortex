`include "VX_platform.vh"

`TRACING_OFF
module VX_multiplier #(
    parameter A_WIDTH = 1,
    parameter B_WIDTH = A_WIDTH,
    parameter R_WIDTH = A_WIDTH + B_WIDTH,
    parameter SIGNED  = 0,
    parameter LATENCY = 0
) (
    input wire clk,    
    input wire enable,
    input wire [A_WIDTH-1:0]  dataa,
    input wire [B_WIDTH-1:0]  datab,
    output wire [R_WIDTH-1:0] result
);
    wire [R_WIDTH-1:0] prod_w;

    if (SIGNED != 0) begin
        assign prod_w = R_WIDTH'($signed(dataa) * $signed(datab));
    end else begin
        assign prod_w = R_WIDTH'(dataa * datab);
    end
    
    if (LATENCY == 0) begin
        assign result = prod_w;
    end else begin        
        reg [R_WIDTH-1:0] prod_r [LATENCY-1:0];
        always @(posedge clk) begin
            if (enable) begin
                prod_r[0] <= prod_w;
                for (integer i = 1; i < LATENCY; ++i) begin
                    prod_r[i] <= prod_r[i-1];
                end
            end
        end        
        assign result = prod_r[LATENCY-1]; 
    end

endmodule
`TRACING_ON

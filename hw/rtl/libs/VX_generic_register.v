`include "VX_define.vh"

module VX_generic_register #( 
    parameter N = 1, 
    parameter PASSTHRU = 0
) (
    input wire          clk,
    input wire          reset,
    input wire          stall,
    input wire          flush,
    input wire[N-1:0]   in,
    output wire[N-1:0]  out
);
    if (PASSTHRU) begin
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (stall)
        assign out = flush ? N'(0) : in;    
    end else begin        
        reg [(N-1):0] value;

        always @(posedge clk) begin
            if (reset) begin
                value <= N'(0);
            end else if (~stall) begin
                value <= in;
            end else if (flush) begin
                value <= N'(0);
            end
        end

        assign out = value;
    end    

endmodule
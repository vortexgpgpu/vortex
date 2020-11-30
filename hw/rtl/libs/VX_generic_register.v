`include "VX_platform.vh"

module VX_generic_register #( 
    parameter N        = 1, 
    parameter R        = N, 
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
        reg [N-1:0] value;

        if (R != 0) begin
            always @(posedge clk) begin
                if (~stall) begin
                    value <= in;
                end
                if (reset || flush) begin
                    value[N-1:N-R] <= R'(0);
                end 
            end
        end else begin
            `UNUSED_VAR (reset)
            `UNUSED_VAR (flush)
            always @(posedge clk) begin
                if (~stall) begin
                    value <= in;
                end
            end
        end

        assign out = value;
    end    

endmodule
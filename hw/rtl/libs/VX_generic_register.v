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
    reg [(N-1):0] value;

    always @(posedge clk) begin
        if (reset) begin
            value <= 0;
        end else if (flush) begin
            value <= 0;
        end else if (~stall) begin
            value <= in;
        end
    end

    assign out = PASSTHRU ? in : value;

endmodule
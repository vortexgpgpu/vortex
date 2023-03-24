`include "VX_platform.vh"

`TRACING_OFF
module VX_tex_lerp (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [7:0]  in1,
    input wire [7:0]  in2,
    input wire [7:0]  frac,
    output wire [7:0] out
);
    `UNUSED_VAR (reset)
    
    reg [15:0] p1, p2;
    reg [15:0] sum;
    reg [7:0]  res;

    wire [7:0] sub = (8'hff - frac);
    
    always @(posedge clk) begin
        if (enable) begin
            p1  <= in1 * sub;
            p2  <= in2 * frac;
            sum <= p1 + p2 + 16'h80;
            res <= 8'((sum + (sum >> 8)) >> 8);
        end
    end

    assign out = res;

endmodule
`TRACING_ON

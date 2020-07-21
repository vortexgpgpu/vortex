
module VX_generic_stack #(
    parameter WIDTH = 1,
    parameter DEPTH = 1
) (
    input  wire              clk,
    input  wire              reset,
    input  wire              push,
    input  wire              pop,
    input  reg [WIDTH - 1:0] q1,
    input  reg [WIDTH - 1:0] q2,
    output wire[WIDTH - 1:0] d
);

    reg [DEPTH - 1:0] ptr;
    reg [WIDTH - 1:0] stack [0:(1 << DEPTH) - 1];

    always @(posedge clk) begin
        if (reset) begin
            ptr <= 0;
        end else if (push) begin
            stack[ptr]   <= q1;
            stack[ptr+1] <= q2;
            ptr          <= ptr + 2;
        end else if (pop) begin
            ptr <= ptr - 1;
        end
    end

    assign d = stack[ptr - 1];

endmodule
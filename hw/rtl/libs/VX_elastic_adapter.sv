`include "VX_platform.vh"

`TRACING_OFF
module VX_elastic_adapter (
    input wire  clk,
    input wire  reset,

    input wire  valid_in,
    output wire ready_in,
        
    input wire  ready_out,
    output wire valid_out,

    input wire  busy,
    output wire strobe
);
    wire push = valid_in && ready_in;
    wire pop = valid_out && ready_out;

    reg loaded;

    always @(posedge clk) begin
        if (reset) begin
            loaded  <= 0;
        end else begin
            if (push) begin
                loaded <= 1;
            end
            if (pop) begin                
                loaded <= 0;
            end
        end
    end

    assign ready_in  = ~loaded;
    assign valid_out = loaded && ~busy;
    assign strobe    = push;

endmodule
`TRACING_ON

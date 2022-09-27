`include "VX_platform.vh"

`TRACING_OFF
module VX_reset_relay #(
    parameter N          = 1,
    parameter MAX_FANOUT = 0
) (
    input wire          clk,
    input wire          reset,
    output wire [N-1:0] reset_o
);    
    if (MAX_FANOUT >= 0 && N > MAX_FANOUT) begin
        localparam F = `UP(MAX_FANOUT);
        localparam R = N / F;
        `PRESERVE_NET reg [R-1:0] reset_r;
        always @(posedge clk) begin
            reset_r <= {R{reset}};
        end
        for (genvar i = 0; i < N; ++i) begin
            assign reset_o[i] = reset_r[i / F];
        end
    end else begin
        `UNUSED_VAR (clk)
        assign reset_o = {N{reset}};
    end
  
endmodule
`TRACING_ON

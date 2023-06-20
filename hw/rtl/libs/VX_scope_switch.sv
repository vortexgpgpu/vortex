`include "VX_platform.vh"

`TRACING_OFF
module VX_scope_switch #(     
    parameter N = 0
) ( 
    input wire  clk,
    input wire  reset,
    input wire  req_in,
    output wire req_out [N],
    input wire  rsp_in [N],
    output wire rsp_out
);
    if (N > 1) begin
        reg req_out_r [N];
        reg rsp_out_r;

        always @(posedge clk) begin
            if (reset) begin
                for (integer i = 0; i < N; ++i) begin
                    req_out_r[i] <= 0;
                end
                rsp_out_r <= 0;
            end else begin            
                for (integer i = 0; i < N; ++i) begin
                    req_out_r[i] <= req_in;
                end
                rsp_out_r <= 0;
                for (integer i = 0; i < N; ++i) begin
                    if (rsp_in[i])
                        rsp_out_r <= 1;
                end
            end
        end

        assign req_out = req_out_r;
        assign rsp_out = rsp_out_r;
    
    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign req_out[0] = req_in;
        assign rsp_out = rsp_in[0];

    end

endmodule
`TRACING_ON

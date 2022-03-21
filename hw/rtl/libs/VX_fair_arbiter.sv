`include "VX_platform.vh"

`TRACING_OFF
module VX_fair_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter LOG_NUM_REQS = $clog2(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     enable,
    input  wire [NUM_REQS-1:0]      requests, 
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
  );

    if (NUM_REQS == 1)  begin                
        
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin    

        reg [NUM_REQS-1:0] buffer;
        reg use_buffer;     

        wire [NUM_REQS-1:0] requests_qual = use_buffer ? buffer : requests;
        wire [NUM_REQS-1:0] buffer_n = requests_qual & ~grant_onehot;

        always @(posedge clk) begin
            if (reset) begin
                use_buffer <= 0;
            end else if (!LOCK_ENABLE || enable) begin
                use_buffer <= (buffer_n != 0);
            end
            if (!LOCK_ENABLE || enable) begin
                buffer <= buffer_n;
            end
        end
               
        VX_fixed_arbiter #(
            .NUM_REQS    (NUM_REQS),
            .LOCK_ENABLE (LOCK_ENABLE)
        ) fixed_arbiter (
            .clk          (clk),
            .reset        (reset),
            .enable       (enable),
            .requests     (requests_qual), 
            .grant_index  (grant_index),
            .grant_onehot (grant_onehot),
            .grant_valid  (grant_valid)
        );
    end
    
endmodule
`TRACING_ON

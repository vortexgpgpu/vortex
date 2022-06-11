`include "VX_platform.vh"

`TRACING_OFF
module VX_cyclic_arbiter #(
    parameter NUM_REQS    = 1,
    parameter LOCK_ENABLE = 0,
    localparam LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [NUM_REQS-1:0]      requests,           
    input  wire                     unlock,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
);
    `UNUSED_PARAM (LOCK_ENABLE)
    `UNUSED_VAR (unlock)
    
    if (NUM_REQS == 1)  begin  

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)      
        
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin

        localparam IS_POW2 = (1 << LOG_NUM_REQS) == NUM_REQS;

        reg [LOG_NUM_REQS-1:0] ctr_r;

        always @(posedge clk) begin
            if (reset) begin
                ctr_r <= 0;
            end else begin                
                ctr_r <= ctr_r + LOG_NUM_REQS'(1);
                if (IS_POW2 && ctr_r == LOG_NUM_REQS'(NUM_REQS-1)) begin
                    ctr_r <= 0;
                end
            end
        end

        assign grant_index = ctr_r;
        
        VX_priority_encoder #(
            .N (NUM_REQS)
        ) priority_encoder (
            .data_in   (requests),
            .index     (grant_index),
            .onehot    (grant_onehot),
            .valid_out (grant_valid)
        );        

    end
    
endmodule
`TRACING_ON

`include "VX_platform.vh"

`TRACING_OFF
module VX_fixed_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter LOG_NUM_REQS = $clog2(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [NUM_REQS-1:0]      requests,           
    input  wire                     enable,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
  );

    `UNUSED_PARAM (LOCK_ENABLE)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (enable)

    if (NUM_REQS == 1)  begin        
        
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin

        VX_priority_encoder #(
            .N (NUM_REQS)
        ) tid_select (
            .data_in   (requests),
            .index     (grant_index),
            .onehot    (grant_onehot),
            .valid_out (grant_valid)
        );

    end
    
endmodule
`TRACING_ON

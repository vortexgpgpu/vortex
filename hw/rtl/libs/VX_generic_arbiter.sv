`include "VX_platform.vh"

`TRACING_OFF
module VX_generic_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter LOG_NUM_REQS = $clog2(NUM_REQS),    
    parameter string TYPE  = "P"
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     enable,
    input  wire [NUM_REQS-1:0]      requests,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
  );

    if (TYPE == "P") begin
        VX_fixed_arbiter #(
            .NUM_REQS    (NUM_REQS),
            .LOCK_ENABLE (LOCK_ENABLE)
        ) grant_arb (
            .clk          (clk),
            .reset        (reset),
            .enable       (enable),
            .requests     (requests),              
            .grant_valid  (grant_valid),
            .grant_index  (grant_index),
            .grant_onehot (grant_onehot)
        );
    end else if (TYPE == "R") begin
        VX_rr_arbiter #(
            .NUM_REQS    (NUM_REQS),
            .LOCK_ENABLE (LOCK_ENABLE)
        ) grant_arb (
            .clk          (clk),
            .reset        (reset),
            .enable       (enable),
            .requests     (requests),  
            .grant_valid  (grant_valid),
            .grant_index  (grant_index),
            .grant_onehot (grant_onehot)
        );
    end else if (TYPE == "F") begin
        VX_fair_arbiter #(
            .NUM_REQS    (NUM_REQS),
            .LOCK_ENABLE (LOCK_ENABLE)
        ) grant_arb (
            .clk          (clk),
            .reset        (reset),
            .enable       (enable),
            .requests     (requests),   
            .grant_valid  (grant_valid),
            .grant_index  (grant_index),
            .grant_onehot (grant_onehot)
        );
    end else if (TYPE == "M") begin
        VX_matrix_arbiter #(
            .NUM_REQS    (NUM_REQS),
            .LOCK_ENABLE (LOCK_ENABLE)
        ) grant_arb (
            .clk          (clk),
            .reset        (reset),
            .enable       (enable),
            .requests     (requests), 
            .grant_valid  (grant_valid),
            .grant_index  (grant_index),
            .grant_onehot (grant_onehot)
        );
    end else begin
        `ERROR(("invalid parameter"));
    end
    
endmodule
`TRACING_ON

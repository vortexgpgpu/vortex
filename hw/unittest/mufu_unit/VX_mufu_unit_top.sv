`include "VX_define.vh"

module VX_mufu_unit_top import VX_gpu_pkg::*; (
    input  wire                            clk,
    input  wire                            reset,

    input  wire                            execute_valid,
    output wire                            execute_ready,
    input  wire [$bits(sfu_execute_t)-1:0] execute_data,

    output wire                            result_valid,
    input  wire                            result_ready,
    output wire [$bits(sfu_result_t)-1:0]  result_data
);
    VX_execute_if #(
        .data_t (sfu_execute_t)
    ) execute_if();

    VX_result_if #(
        .data_t (sfu_result_t)
    ) result_if();

    assign execute_if.valid = execute_valid;
    assign execute_if.data  = execute_data;
    assign execute_ready    = execute_if.ready;

    assign result_valid     = result_if.valid;
    assign result_data      = result_if.data;
    assign result_if.ready  = result_ready;

    VX_mufu_unit #(
        .INSTANCE_ID ("mufu_unit_top"),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES)
    ) mufu_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (execute_if),
        .result_if  (result_if)
    );

endmodule

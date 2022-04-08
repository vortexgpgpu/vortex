`include "VX_rop_define.vh"

import VX_rop_types::*;

module VX_rop_unit #(    
    parameter CLUSTER_ID = 0,    
    parameter NUM_SLICES = 1,
    parameter NUM_LANES  = NUM_SLICES
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_rop_perf_if.master rop_perf_if,
`endif

    // Memory interface
    VX_cache_req_if.master cache_req_if,
    VX_cache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_rop_dcr_if.slave rop_dcr_if,
    VX_rop_req_if.slave rop_req_if
);
    rop_dcrs_t dcrs = rop_dcr_if.data;

    VX_rop_slice #(
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_LANES  (NUM_LANES)
    ) rop_slice (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .rop_perf_if    (rop_perf_if),
    `endif
        .dcrs           (dcrs),

        .cache_req_if   (cache_req_if),
        .cache_rsp_if   (cache_rsp_if),

        .rop_req_if     (rop_req_if)
    );

endmodule

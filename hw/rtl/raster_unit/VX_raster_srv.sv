`include "VX_raster_define.vh"

module VX_raster_srv #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs    
    VX_raster_req_if.slave  raster_req_if,
    VX_gpu_csr_if.slave     raster_csr_if,
    
    // Outputs
    VX_commit_if.master     raster_rsp_if
);

    // TODO
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

endmodule
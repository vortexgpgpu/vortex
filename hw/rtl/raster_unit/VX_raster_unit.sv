`include "VX_raster_define.vh"

module VX_raster_unit #(  
    parameter CORE_ID = 0
    // TODO
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_perf_tex_if.master perf_raster_if,
`endif

    // Memory interface
    VX_dcache_req_if.master cache_req_if,
    VX_dcache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_raster_csr_if.slave  raster_csr_if,
    VX_raster_req_if.slave  raster_req_if,

    // Outputs
    VX_raster_rsp_if.master raster_rsp_if
);

    // TODO

endmodule
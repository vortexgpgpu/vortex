`include "VX_raster_define.vh"

module VX_raster_csr #(  
    parameter CORE_ID = 0
    // TODO
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_raster_csr_if.slave raster_csr_if,
    VX_raster_req_if.slave raster_req_if,

    // Output
    output raster_csrs_t raster_csrs
);

    // TODO

endmodule
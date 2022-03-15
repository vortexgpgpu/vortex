`include "VX_raster_define.vh"

// Module for handling memory requests
//  Descrption: Performs strided fetch
//  of primitive data from the buffer

module VX_raster_mem #( 
    parameter CLUSTER_ID = 0,
    parameter RASTER_SLICE_NUM = 1,
    parameter RASTER_PRIM_REQUEST_SIZE = 5,
) (
    input logic clk,
    input logic reset,
);

    // TODO
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

endmodule
`include "VX_raster_define.vh"

// Module for primitive fetch
//  Descrption: Performs strided fetch
//  of primitive data from the buffer

module VX_raster_fetch #(  
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
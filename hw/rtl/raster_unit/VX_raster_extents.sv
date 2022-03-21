`include "VX_raster_define.vh"

module VX_raster_extents #(
    parameter RASTER_TILE_SIZE = 64
) (
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edges[2:0],
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  extents
);
    assign extents = 
        ((edges[0] >= 0 ? edges[0] : {`RASTER_PRIMITIVE_DATA_BITS{1'b0}})
        +
        (edges[1] >= 0 ? edges[1] : {`RASTER_PRIMITIVE_DATA_BITS{1'b0}}))
        << $clog2(RASTER_TILE_SIZE);
endmodule

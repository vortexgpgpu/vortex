`include "VX_raster_define.vh"

module VX_raster_extents #(
    parameter TILE_LOGSIZE = 64
) (
    input wire signed [2:0][2:0][`RASTER_DATA_BITS-1:0] edges,
    output wire signed [2:0][`RASTER_DATA_BITS-1:0]     extents
);
    for (genvar i = 0; i < 3; ++i) begin
        assign extents[i] = (edges[i][0] >= 0 ? (edges[i][0] << TILE_LOGSIZE) : `RASTER_DATA_BITS'(0))
                          + (edges[i][1] >= 0 ? (edges[i][1] << TILE_LOGSIZE) : `RASTER_DATA_BITS'(0));
    end

endmodule

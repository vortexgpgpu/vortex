`include "VX_raster_define.vh"

module VX_raster_extents #(
    parameter TILE_LOGSIZE = 5
) (
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges,
    output wire [2:0][`RASTER_DATA_BITS-1:0]     extents
);
    for (genvar i = 0; i < 3; ++i) begin
        wire [`RASTER_DATA_BITS-1:0] edge_x_m = {`RASTER_DATA_BITS{~edges[i][0][`RASTER_DATA_BITS-1]}};
        wire [`RASTER_DATA_BITS-1:0] edge_y_m = {`RASTER_DATA_BITS{~edges[i][1][`RASTER_DATA_BITS-1]}};
        assign extents[i] = (edge_x_m & (edges[i][0] << TILE_LOGSIZE))
                          + (edge_y_m & (edges[i][1] << TILE_LOGSIZE));
    end

endmodule

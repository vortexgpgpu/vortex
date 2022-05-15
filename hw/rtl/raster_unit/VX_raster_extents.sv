`include "VX_raster_define.vh"

module VX_raster_extents #(
    parameter TILE_SIZE = 64
) (
    input wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  edges[2:0][2:0],
    output wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] extents[2:0]
);
    localparam TILE_BITS = $clog2(TILE_SIZE);

    for (genvar i = 0; i < 3; ++i) begin
        assign extents[i] = (edges[i][0] >= 0 ? (edges[i][0] << TILE_BITS): `RASTER_PRIMITIVE_DATA_BITS'(0))
                          + (edges[i][1] >= 0 ? (edges[i][1] << TILE_BITS) : `RASTER_PRIMITIVE_DATA_BITS'(0));
    end

endmodule

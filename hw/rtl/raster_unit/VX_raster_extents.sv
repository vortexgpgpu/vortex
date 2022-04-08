`include "VX_raster_define.vh"

import VX_raster_types::*;

module VX_raster_extents #(
    parameter RASTER_TILE_SIZE = 64
) (
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edges[2:0][2:0],
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  extents[2:0]
);
    for (genvar i = 0; i < 3; ++i) begin
        assign extents[i] = 
            ((edges[i][0] >= 0 ? edges[i][0] : {`RASTER_PRIMITIVE_DATA_BITS{1'b0}})
            +
            (edges[i][1] >= 0 ? edges[i][1] : {`RASTER_PRIMITIVE_DATA_BITS{1'b0}}))
            << $clog2(RASTER_TILE_SIZE);
    end

// `ifdef DBG_TRACE_CORE_PIPELINE
//     always @(posedge clk) begin
//         for (int i = 0; i < 3; ++i) begin
//             dpi_trace(1, "Extents: %d: i=%d , val = %d\n", 
//                 $time, i, extents[i]);
//         end
//     end
// `endif
endmodule

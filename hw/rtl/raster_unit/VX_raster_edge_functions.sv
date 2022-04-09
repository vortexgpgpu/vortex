// Module to evaluate the edge function

`include "VX_raster_define.vh"

module VX_raster_edge_functions (
    // input logic clk,
    input logic [`RASTER_DIM_BITS-1:0]  x_loc, y_loc,
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] edges[2:0][2:0],
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] edge_func_val[2:0]
);
    for (genvar i = 0; i < 3; ++i) begin
        logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] val1 = (edges[i][0] * x_loc) >>> 16;
        logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] val2 = (edges[i][1] * y_loc) >>> 16;
        always_comb begin
            edge_func_val[i] = val1 + val2 + edges[i][2];
        end
    end

// `ifdef DBG_TRACE_CORE_PIPELINE
//     always @(posedge clk) begin
//         for (int i = 0; i < 3; ++i) begin
//             dpi_trace(1, "Edge func eval: %d: x_loc = %0d, y_loc = %0d, i=%d , a = %d, b = %d, c = %d, val = %d\n", 
//                 $time, x_loc, y_loc, i, edges[i][0], edges[i][1], edges[i][2], edge_func_val[i]);
//         end
//     end
// `endif

endmodule
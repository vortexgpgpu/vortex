// Module to evaluate the edge function

module VX_raster_edge_functions (
    input logic [`RASTER_DIM_BITS-1:0]  x_loc, y_loc,
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] edges[2:0][2:0],
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] edge_func_val[2:0]
);
    for (genvar i = 0; i < 3; ++i) begin
        always_comb begin
            edge_func_val[i] = edges[i][0] * x_loc + edges[i][1] * y_loc + edges[i][2];
        end
    end

endmodule;
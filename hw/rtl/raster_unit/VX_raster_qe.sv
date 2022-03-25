// Quad evaluator block
// Functionality: Receives a 2x2 quad with primitive information
//     check whether quad pixels are within the primitive

`include "VX_raster_define.vh"

module VX_raster_qe (
    // Primitive related data
    // edge equation data for the 3 edges and ax+by+c
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]       edges[2:0][2:0],
    // edge function computation value propagated
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]       edge_func_val[2:0],

    // Mask bits for the 2x2 quad
    output logic [3:0] masks,
    // barycentric coordinates
    output logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]             bcoords[2:0][3:0] // dim1 => quad index
);

    // New edge value for all 4 pixels (0,0) (0,1) (1,0) (1,1)
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] new_edge_val [2:0][1:0][1:0];

    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
                always_comb begin
                    integer k;
                    for (k = 0; k < 3; ++k)
                        new_edge_val[k][i][j] = edge_func_val[k] + i*edges[k][0] + j*edges[k][1];
                    masks[i*2 + j] = 0;
                    if (new_edge_val[0][i][j] >= 0 && new_edge_val[1][i][j] >= 0 && new_edge_val[2][i][j] >= 0) begin
                        masks[i*2 + j] = 1;
                        bcoords[0][i*2 + j] = new_edge_val[0][i][j];
                        bcoords[1][i*2 + j] = new_edge_val[1][i][j];
                        bcoords[2][i*2 + j] = new_edge_val[2][i][j];
                    end
                end
            end
        end

endmodule

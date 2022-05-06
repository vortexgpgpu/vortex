// Quad evaluator block
// Functionality: Receives a 2x2 quad with primitive information
//     check whether quad pixels are within the primitive

`include "VX_raster_define.vh"

module VX_raster_qe #(
    parameter SLICE_ID  = 1,
    parameter QUAD_ID   = 1
) (
    input logic clk, reset, enable,
    // Quad data
    input logic         [`RASTER_DIM_BITS-1:0]                  x_loc, y_loc,
    // Primitive related data
    // edge equation data for the 3 edges and ax+by+c
    input logic signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]       edges[2:0][2:0],
    // edge function computation value propagated
    input logic signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]       edge_func_val[2:0],

    // Rendering region
    input logic         [`RASTER_DIM_BITS-1:0]                  dst_width, dst_height,

    // Mask bits for the 2x2 quad
    output logic        [3:0]                                   masks,
    // barycentric coordinates
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]       bcoords[2:0][3:0] // dim1 => quad index
);

    // New edge value for all 4 pixels (0,0) (0,1) (1,0) (1,1)
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] new_edge_val [2:0][1:0][1:0];

    // Generate new_edge_val
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            for (genvar k = 0; k < 3; ++k) begin
                assign new_edge_val[k][i][j] = edge_func_val[k] + i*edges[k][0] + j*edges[k][1];
            end
        end
    end

    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] new_edge_val_r [2:0][1:0][1:0];
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            for (genvar k = 0; k < 3; ++k) begin
                VX_pipe_register #(
                    .DATAW  (`RASTER_PRIMITIVE_DATA_BITS),
                    .RESETW (1)
                ) qe_pipe_reg_2 (
                    .clk      (clk),
                    .reset    (reset),
                    .enable   (enable),
                    .data_in  (new_edge_val[k][i][j]),
                    .data_out (new_edge_val_r[k][i][j])
                );
            end
        end
    end


    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            always_comb begin
                masks[j*2 + i] = 0;
                bcoords[0][j*2 + i] = 0;
                bcoords[1][j*2 + i] = 0;
                bcoords[2][j*2 + i] = 0;
                if (new_edge_val_r[0][i][j] >= 0 && new_edge_val_r[1][i][j] >= 0 && new_edge_val_r[2][i][j] >= 0) begin
                    if (((x_loc >> 1) + i) < dst_width && ((y_loc >> 1) + j) < dst_height) begin
                        masks[j*2 + i] = 1;
                        bcoords[0][j*2 + i] = new_edge_val_r[0][i][j];
                        bcoords[1][j*2 + i] = new_edge_val_r[1][i][j];
                        bcoords[2][j*2 + i] = new_edge_val_r[2][i][j];
                    end
                end
            end
        end
    end

endmodule

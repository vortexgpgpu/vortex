// Quad evaluator block
// Functionality: Receives a 2x2 quad with primitive information
//     check whether quad pixels are within the primitive

`include "VX_raster_define.vh"

module VX_raster_qe #(
    parameter SLICE_ID = 1,
    parameter QUAD_ID  = 1
) (
    input wire clk,
    input wire reset, 
    
    input wire enable,

    // Quad data
    input wire         [`RASTER_DIM_BITS-1:0]                  x_loc,
    input wire         [`RASTER_DIM_BITS-1:0]                  y_loc,
    // Primitive related data
    // edge equation data for the 3 edges and ax+by+c
    input wire signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]       edges[2:0][2:0],
    // edge function computation value propagated
    input wire signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]       edge_func_val[2:0],

    // Rendering region
    input wire         [`RASTER_DIM_BITS-1:0]                  dst_width,
    input wire         [`RASTER_DIM_BITS-1:0]                  dst_height,

    input wire                                                 out_enable,
    // Output of piped x_loc, y_loc
    output wire        [`RASTER_DIM_BITS-1:0]                  x_loc_o,
    output wire        [`RASTER_DIM_BITS-1:0]                  y_loc_o,
    // Mask bits for the 2x2 quad
    output wire        [3:0]                                   masks_o,
    // barycentric coordinates
    output wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]       bcoords_o[2:0][3:0]
);

    // New edge value for all 4 pixels (0,0) (0,1) (1,0) (1,1)
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] new_edge_val [2:0][1:0][1:0];

    // Generate new_edge_val
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            for (genvar k = 0; k < 3; ++k) begin
                assign new_edge_val[k][i][j] = edge_func_val[k] + i*edges[k][0] + j*edges[k][1];
            end
        end
    end

    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] new_edge_val_r [2:0][1:0][1:0];

    wire [3*2*2*`RASTER_PRIMITIVE_DATA_BITS-1:0] pipe_reg_2_in, pipe_reg_2_r;
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            for (genvar k = 0; k < 3; ++k) begin
                assign pipe_reg_2_in[((i*2+j)*3+k)*`RASTER_PRIMITIVE_DATA_BITS+:`RASTER_PRIMITIVE_DATA_BITS] = new_edge_val[k][i][j];
                assign new_edge_val_r[k][i][j] = pipe_reg_2_r[((i*2+j)*3+k)*`RASTER_PRIMITIVE_DATA_BITS+:`RASTER_PRIMITIVE_DATA_BITS];
            end
        end
    end

    wire        [`RASTER_DIM_BITS-1:0]                  x_loc_r, y_loc_r;
    VX_pipe_register #(
        .DATAW  (2*`RASTER_DIM_BITS + 3*2*2*`RASTER_PRIMITIVE_DATA_BITS),
        .RESETW (1)
    ) qe_pipe_reg_2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({x_loc, y_loc, pipe_reg_2_in}),
        .data_out ({x_loc_r, y_loc_r, pipe_reg_2_r})
    );

    // Mask bits for the 2x2 quad
    reg        [3:0]                                   masks;
    // barycentric coordinates
    reg signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]       bcoords[2:0][3:0]; // dim1 => quad index
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

    VX_pipe_register #(
        .DATAW  (4 + 2*`RASTER_DIM_BITS + 3*4*`RASTER_PRIMITIVE_DATA_BITS),
        .RESETW (4)
    ) qe_pipe_reg_out (
        .clk      (clk),
        .reset    (reset),
        .enable   (out_enable),
        .data_in  ({masks, x_loc_r, y_loc_r,
            bcoords[0][0], bcoords[0][1], bcoords[0][2], bcoords[0][3],
            bcoords[1][0], bcoords[1][1], bcoords[1][2], bcoords[1][3],
            bcoords[2][0], bcoords[2][1], bcoords[2][2], bcoords[2][3]
        }),
        .data_out ({masks_o, x_loc_o, y_loc_o,
            bcoords_o[0][0], bcoords_o[0][1], bcoords_o[0][2], bcoords_o[0][3],
            bcoords_o[1][0], bcoords_o[1][1], bcoords_o[1][2], bcoords_o[1][3],
            bcoords_o[2][0], bcoords_o[2][1], bcoords_o[2][2], bcoords_o[2][3]
            })
    );

endmodule

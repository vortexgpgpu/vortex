// Tile evaluator block
// Functionality: Evaluates the input tile to check:
//     1. If it is valid => overlaps triangle
//     2. If it is a block
//     3. Else divides it into 4

`include "VX_raster_define.vh"

module VX_raster_te #(
    parameter RASTER_TILE_SIZE       = 16,
    parameter RASTER_BLOCK_SIZE      = 4,
    parameter RASTER_LEVEL_DATA_BITS = ($clog2(RASTER_TILE_SIZE/RASTER_BLOCK_SIZE) + 1)
) (
    // Level value in recursive descent
    input logic [RASTER_LEVEL_DATA_BITS-1:0]                 level,
    // Tile data
    input logic [`RASTER_TILE_DATA_BITS-1:0]                  x_loc, y_loc,
    // Primitive data
    // edge equation data for the 3 edges and ax+by+c
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]      edges[2:0][2:0],
    // edge function computation value propagated
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]      edge_func_val[2:0],
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]      extents[2:0],

    // Status signals
    output logic                                              valid_tile, valid_block,
    // Sub-tile related data
    output logic [`RASTER_TILE_DATA_BITS-1:0]                 tile_x_loc[3:0],
        tile_y_loc[3:0],
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]     tile_edge_func_val[3:0][2:0]
);
    localparam RASTER_TILE_FIFO_DEPTH    = RASTER_TILE_SIZE/RASTER_BLOCK_SIZE;
    localparam RASTER_TILE_SIZE_BITS     = $clog2(RASTER_TILE_SIZE);
    localparam RASTER_BLOCK_SIZE_BITS    = $clog2(RASTER_BLOCK_SIZE);

    // Check if primitive within tile
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] eval0, eval1, eval2;
    assign eval0 = (edge_func_val[0] + extents[0]) >> level;
    assign eval1 = (edge_func_val[1] + extents[1]) >> level;
    assign eval2 = (edge_func_val[2] + extents[2]) >> level;

    // Sub-tile specs info
    logic [`RASTER_TILE_DATA_BITS-1:0] sub_tile_size;
    logic [`RASTER_TILE_DATA_BITS-1:0] sub_tile_bits;

    always_comb begin
        // Check if tile has triangle
        valid_tile = !((eval0 < 0) || (eval1 < 0) || (eval2 < 9));
        // If tile valid => sub-divide into sub-tiles
        if (valid_tile) begin
            sub_tile_bits = `RASTER_TILE_DATA_BITS'(RASTER_TILE_SIZE_BITS) - `RASTER_TILE_DATA_BITS'(level) - 1;
            sub_tile_size = 1 << sub_tile_bits;
            if (sub_tile_bits >= `RASTER_TILE_DATA_BITS'(RASTER_BLOCK_SIZE_BITS)) begin
                // divide into sub tiles as still bigger than block
                valid_block = 0;
                // generate sub-tile data
                for (integer i = 0; i < 2; ++i) begin
                    for (integer j = 0; j < 2; ++j) begin
                        tile_x_loc[i*2+j] = x_loc + `RASTER_TILE_DATA_BITS'(i)*sub_tile_size;
                        tile_y_loc[i*2+j] = y_loc + `RASTER_TILE_DATA_BITS'(j)*sub_tile_size;
                        for (integer k = 0; k < 3; ++k) begin
                            tile_edge_func_val[i*2+j][k] = edge_func_val[k]
                                + i*(edges[k][0] << sub_tile_bits)
                                + j*(edges[k][1] << sub_tile_bits);
                        end
                    end
                end
            end
            else begin
                // run block evaluator on valid block
                valid_block = 1;
                // Deassert valid_tile so that it tells whether it generated a block or tile or neither
                valid_tile = 0;
            end
        end
    end
endmodule

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
    input logic clk, reset, stall,
    // Input valid
    input logic input_valid,
    // Level value in recursive descent
    input logic        [RASTER_LEVEL_DATA_BITS-1:0]         level,
    // Tile data
    input logic        [`RASTER_DIM_BITS-1:0]               x_loc, y_loc,
    // Primitive data
    // edge equation data for the 3 edges and ax+by+c
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    edges[2:0][2:0],
    // edge function computation value propagated
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    edge_func_val[2:0],
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    extents[2:0],

    // Status signals
    output logic                                            valid_tile, valid_block,
    // Sub-tile related data
    output logic        [`RASTER_DIM_BITS-1:0]              tile_x_loc[3:0],
                                                            tile_y_loc[3:0],
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   tile_edge_func_val[3:0][2:0],
    // Block related data
    output logic        [`RASTER_DIM_BITS-1:0]              block_x_loc,
                                                            block_y_loc,
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   block_edge_func_val[2:0],
    output logic        [RASTER_LEVEL_DATA_BITS-1:0]        tile_level
);

    localparam RASTER_TILE_FIFO_DEPTH    = RASTER_TILE_SIZE/RASTER_BLOCK_SIZE;
    localparam RASTER_TILE_SIZE_BITS     = $clog2(RASTER_TILE_SIZE);
    localparam RASTER_BLOCK_SIZE_BITS    = $clog2(RASTER_BLOCK_SIZE);

    logic input_valid_r;
    logic        [RASTER_LEVEL_DATA_BITS-1:0]         level_r;
    logic        [`RASTER_DIM_BITS-1:0]               x_loc_r, y_loc_r;
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    edges_r[2:0][2:0];
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    edge_func_val_r[2:0];
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    extents_r[2:0];
    VX_pipe_register #(
        .DATAW  (1 + RASTER_LEVEL_DATA_BITS + 2*`RASTER_DIM_BITS + 3*3*`RASTER_PRIMITIVE_DATA_BITS
            + 3*`RASTER_PRIMITIVE_DATA_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS),
        .RESETW (1)
    ) te_pipe_reg_1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({
            input_valid, level, x_loc, y_loc,
            edges[0][0], edges[0][1], edges[0][2],
            edges[1][0], edges[1][1], edges[1][2],
            edges[2][0], edges[2][1], edges[2][2],
            edge_func_val[0], edge_func_val[1], edge_func_val[2],
            extents[0], extents[1], extents[2]
        }),
        .data_out ({
            input_valid_r, level_r, x_loc_r, y_loc_r,
            edges_r[0][0], edges_r[0][1], edges_r[0][2],
            edges_r[1][0], edges_r[1][1], edges_r[1][2],
            edges_r[2][0], edges_r[2][1], edges_r[2][2],
            edge_func_val_r[0], edge_func_val_r[1], edge_func_val_r[2],
            extents_r[0], extents_r[1], extents_r[2]
        })
    );

    // Check if primitive within tile
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] eval0, eval1, eval2;
    assign eval0 = (edge_func_val_r[0] + extents_r[0]) >> level_r;
    assign eval1 = (edge_func_val_r[1] + extents_r[1]) >> level_r;
    assign eval2 = (edge_func_val_r[2] + extents_r[2]) >> level_r;

    // Sub-tile specs info
    logic [`RASTER_DIM_BITS-1:0] sub_tile_size;
    logic [`RASTER_DIM_BITS-1:0] sub_tile_bits;

    // Generate the x,y loc and edge function values
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            assign tile_x_loc[i*2+j] = x_loc_r + `RASTER_DIM_BITS'(i)*sub_tile_size;
            assign tile_y_loc[i*2+j] = y_loc_r + `RASTER_DIM_BITS'(j)*sub_tile_size;
        end 
    end
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            for (genvar k = 0; k < 3; ++k) begin
                assign tile_edge_func_val[i*2+j][k] = edge_func_val_r[k]
                    + i*(edges_r[k][0] << sub_tile_bits)
                    + j*(edges_r[k][1] << sub_tile_bits);
            end
        end
    end

    always_comb begin
        valid_block = 0;
        // Check if tile has triangle
        valid_tile = (!((eval0 < 0) || (eval1 < 0) || (eval2 < 0))) & input_valid_r;
        // If tile valid => sub-divide into sub-tiles
        sub_tile_bits = `RASTER_DIM_BITS'(RASTER_TILE_SIZE_BITS) - `RASTER_DIM_BITS'(level_r) - `RASTER_DIM_BITS'(1);
        sub_tile_size = `RASTER_DIM_BITS'(1) << sub_tile_bits;
        if (valid_tile) begin
            if (!(sub_tile_bits >= `RASTER_DIM_BITS'(RASTER_BLOCK_SIZE_BITS))) begin
                // run block evaluator on valid block
                valid_block = 1;
                // Deassert valid_tile so that it tells whether it generated a block or tile or neither
                valid_tile = 0;
            end
        end
    end

    assign tile_level = level_r;
    assign block_x_loc = x_loc_r;
    assign block_y_loc = y_loc_r;
    assign block_edge_func_val = edge_func_val_r;


`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (input_valid) begin
            dpi_trace(2, "%d: raster-tile-in: level=%0d, x=%0d, y=%0d, edge1.x=%0d, edge1.y=%0d, edge1.z=%0d, edge2.x=%0d, edge2.y=%0d, edge2.z=%0d, edge3.x=%0d, edge3.y=%0d, edge3.z=%0d, edge_func_val=%0d %0d %0d, extents=%0d %0d %0d\n",
                $time, level, x_loc, y_loc,
                edges[0][0], edges[0][1], edges[0][2],
                edges[1][0], edges[1][1], edges[1][2],
                edges[2][0], edges[2][1], edges[2][2],
                edge_func_val[0], edge_func_val[1], edge_func_val[2],
                extents[0], extents[1], extents[2]);
        end
    end
    always @(posedge clk) begin
        if (valid_tile) begin
            for (int i = 0; i < 3; ++i) begin
                dpi_trace(2, "%d: raster-tile-out: valid_tile level=%0d, x=%0d, y=%0d, edge_func_val=%0d %0d %0d\n",
                    $time, tile_level, tile_x_loc[i], tile_y_loc[i],
                    tile_edge_func_val[i][0], tile_edge_func_val[i][1], tile_edge_func_val[i][2]);
            end
        end
        if (valid_block) begin
            dpi_trace(2, "%d: raster-tile-out: valid_block level=%0d, x=%0d, y=%0d, edge_func_val=%0d %0d %0d\n",
                $time, tile_level, block_x_loc, block_y_loc,
                block_edge_func_val[0], block_edge_func_val[1], block_edge_func_val[2]);
        end
    end
`endif

endmodule

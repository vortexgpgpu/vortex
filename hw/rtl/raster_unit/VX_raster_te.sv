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
    output logic                                            tile_valid, block_valid,
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


    // Check if primitive within tile
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] eval0, eval1, eval2;
    assign eval0 = (edge_func_val[0] + extents[0]) >> level;
    assign eval1 = (edge_func_val[1] + extents[1]) >> level;
    assign eval2 = (edge_func_val[2] + extents[2]) >> level;

    // Sub-tile specs info
    logic [`RASTER_DIM_BITS-1:0] sub_tile_size;
    logic [`RASTER_DIM_BITS-1:0] sub_tile_bits;

    // Generate the x,y loc and edge function values
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            assign tile_x_loc_r[i*2+j] = x_loc + `RASTER_DIM_BITS'(i)*sub_tile_size;
            assign tile_y_loc_r[i*2+j] = y_loc + `RASTER_DIM_BITS'(j)*sub_tile_size;
        end 
    end
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            for (genvar k = 0; k < 3; ++k) begin
                assign tile_edge_func_val_r[i*2+j][k] = edge_func_val[k]
                    + i*(edges[k][0] << sub_tile_bits)
                    + j*(edges[k][1] << sub_tile_bits);
            end
        end
    end

    always_comb begin
        block_valid_r = 0;
        // Check if tile has triangle
        tile_valid_r = (!((eval0 < 0) || (eval1 < 0) || (eval2 < 0))) & input_valid;
        // If tile valid => sub-divide into sub-tiles
        sub_tile_bits = `RASTER_DIM_BITS'(RASTER_TILE_SIZE_BITS) - `RASTER_DIM_BITS'(level) - `RASTER_DIM_BITS'(1);
        sub_tile_size = `RASTER_DIM_BITS'(1) << sub_tile_bits;
        if (tile_valid_r) begin
            if (!(sub_tile_bits >= `RASTER_DIM_BITS'(RASTER_BLOCK_SIZE_BITS))) begin
                // run block evaluator on valid block
                block_valid_r = 1;
                // Deassert tile_valid_r so that it tells whether it generated a block or tile or neither
                tile_valid_r = 0;
            end
        end
    end

    assign tile_level_r = level;
    assign block_x_loc_r = x_loc;
    assign block_y_loc_r = y_loc;
    assign block_edge_func_val_r = edge_func_val;

    // Status signals
    logic                                            tile_valid_r, block_valid_r;
    // Sub-tile related data
    logic        [`RASTER_DIM_BITS-1:0]              tile_x_loc_r[3:0],
                                                     tile_y_loc_r[3:0];
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   tile_edge_func_val_r[3:0][2:0];
    // Block related data
    logic        [`RASTER_DIM_BITS-1:0]              block_x_loc_r,
                                                     block_y_loc_r;
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   block_edge_func_val_r[2:0];
    logic        [RASTER_LEVEL_DATA_BITS-1:0]        tile_level_r;


    VX_pipe_register #(
        .DATAW  (2 + 2*4*`RASTER_DIM_BITS + 4*3*`RASTER_PRIMITIVE_DATA_BITS + 2*`RASTER_DIM_BITS +
            3*`RASTER_PRIMITIVE_DATA_BITS + RASTER_LEVEL_DATA_BITS),
        .RESETW (2)
    ) te_pipe_reg_1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({
            tile_valid_r, block_valid_r,
            tile_x_loc_r[0], tile_x_loc_r[1], tile_x_loc_r[2], tile_x_loc_r[3],
            tile_y_loc_r[0], tile_y_loc_r[1], tile_y_loc_r[2], tile_y_loc_r[3],
            tile_edge_func_val_r[0][0], tile_edge_func_val_r[0][1], tile_edge_func_val_r[0][2],
            tile_edge_func_val_r[1][0], tile_edge_func_val_r[1][1], tile_edge_func_val_r[1][2],
            tile_edge_func_val_r[2][0], tile_edge_func_val_r[2][1], tile_edge_func_val_r[2][2],
            tile_edge_func_val_r[3][0], tile_edge_func_val_r[3][1], tile_edge_func_val_r[3][2],
            block_x_loc_r, block_y_loc_r,
            block_edge_func_val_r[0], block_edge_func_val_r[1], block_edge_func_val_r[2],
            tile_level_r
        }),
        .data_out ({
            tile_valid, block_valid,
            tile_x_loc[0], tile_x_loc[1], tile_x_loc[2], tile_x_loc[3],
            tile_y_loc[0], tile_y_loc[1], tile_y_loc[2], tile_y_loc[3],
            tile_edge_func_val[0][0], tile_edge_func_val[0][1], tile_edge_func_val[0][2],
            tile_edge_func_val[1][0], tile_edge_func_val[1][1], tile_edge_func_val[1][2],
            tile_edge_func_val[2][0], tile_edge_func_val[2][1], tile_edge_func_val[2][2],
            tile_edge_func_val[3][0], tile_edge_func_val[3][1], tile_edge_func_val[3][2],
            block_x_loc, block_y_loc,
            block_edge_func_val[0], block_edge_func_val[1], block_edge_func_val[2],
            tile_level
        })
    );

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
        if (tile_valid) begin
            for (int i = 0; i < 3; ++i) begin
                dpi_trace(2, "%d: raster-tile-out: tile_valid level=%0d, x=%0d, y=%0d, edge_func_val=%0d %0d %0d\n",
                    $time, tile_level, tile_x_loc[i], tile_y_loc[i],
                    tile_edge_func_val[i][0], tile_edge_func_val[i][1], tile_edge_func_val[i][2]);
            end
        end
        if (block_valid) begin
            dpi_trace(2, "%d: raster-tile-out: block_valid level=%0d, x=%0d, y=%0d, edge_func_val=%0d %0d %0d\n",
                $time, tile_level, block_x_loc, block_y_loc,
                block_edge_func_val[0], block_edge_func_val[1], block_edge_func_val[2]);
        end
    end
`endif

endmodule

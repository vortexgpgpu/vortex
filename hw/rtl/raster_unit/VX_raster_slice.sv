// Raster slice
// Functionality:
//     1. Recursive descent
//     2. Tile evaluation
//     3. Quad evaluation and storage
//     4. Return the quad(s)

`include "VX_raster_define.vh"

module VX_raster_slice #(
    parameter RASTER_BLOCK_SIZE       = 8,
    parameter RASTER_TILE_SIZE        = 16,
    parameter RASTER_QUAD_OUTPUT_RATE = 4,
    parameter RASTER_QUAD_FIFO_DEPTH  = 64,
    parameter RASTER_TILE_FIFO_DEPTH  = 16
) (
    // Standard inputs
    input logic                                           clk, reset,
    // To indicate valid input provided
    input logic                                           input_valid,
    // Tile information
    input logic [`RASTER_TILE_DATA_BITS-1:0]              x_loc, y_loc,
    // Primitive information
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  edges[2:0][2:0],
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  edge_func_val[2:0],
    input logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  extents[2:0],

    // Hand-shaking signals
    input logic                                           pop_quad,
    output logic                                          ready, quad_queue_empty,

    // Output sub-tiles data
    output logic [`RASTER_TILE_DATA_BITS-1:0]             out_quad_x_loc[RASTER_QUAD_OUTPUT_RATE-1:0],
        out_quad_y_loc[RASTER_QUAD_OUTPUT_RATE-1:0],
    output logic [3:0]                                    out_quad_masks[RASTER_QUAD_OUTPUT_RATE-1:0],
    output logic                                          valid[RASTER_QUAD_OUTPUT_RATE-1:0]
);

    localparam RASTER_LEVEL_DATA_BITS = $clog2(RASTER_TILE_SIZE/RASTER_BLOCK_SIZE) + 1;
    localparam RASTER_FIFO_DATA_WIDTH = (RASTER_LEVEL_DATA_BITS + 2*`RASTER_TILE_DATA_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS);

    // Store data which will stay same for tile throughout operation
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  global_extents[2:0];
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  global_edges[2:0][2:0];

    // Store the tile relevant data as global regs as TE is combinatorial
    logic [`RASTER_TILE_DATA_BITS-1:0]              tile_x_loc, tile_y_loc;
    logic [RASTER_LEVEL_DATA_BITS-1:0]              level;
    logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  tile_edge_func_val[2:0];

    logic [RASTER_LEVEL_DATA_BITS-1:0] level_1;
    assign level_1 = level + 1;

    // Control signsl
    logic        valid_tile, valid_block;
    logic        fifo_full, fifo_empty, fifo_tile_valid;
    logic        stall;
    logic        be_ready; // to track the status of the block evaluator
    logic       done;

    // Stall used to wait for block queue to complete run if another needs to be inserted
    assign stall = (valid_block == 1 && block_fifo_full == 1);

    // Incoming tile data from fifo
    logic [`RASTER_TILE_DATA_BITS-1:0]               fifo_tile_x_loc, fifo_tile_y_loc;
    logic [RASTER_LEVEL_DATA_BITS-1:0]              fifo_tile_level;
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]          fifo_tile_edge_func_val[2:0];

    // Tile data selector to choose tile data from:
    //     1. Input tile
    //     2. Sub-tile for queue
    //     3. Data forwarded sub-tile for first iteration when
    //        input tile divided into 4, but sub-tile fifo will
    //        be empty. So, instead of wasting 1 cycle, sub=tile
    //        forwarded directly here.
    always @(posedge clk) begin
        if (reset) begin
            done <= 0;
            // Reset all globals and signals
            for (integer i = 0; i < 3; ++i) begin
                global_extents[i] <= 0;
                tile_edge_func_val[i] <= 0;
                for (integer j = 0; j < 3; ++j) begin
                    global_edges[i][j] <= 0;
                end
            end
        end
        else if (stall == 0) begin
            // if block ready and input valid read data from input
            if (ready == 1 && input_valid == 1) begin
                tile_x_loc         <= x_loc;
                tile_y_loc         <= y_loc;
                level              <= 0;
                tile_edge_func_val <= edge_func_val;
                // Update globals
                global_extents     <= extents;
                global_edges       <= edges;
            end
            // sub-tile rerouter used only 1 onces for the parent tile
            else if (level == 0 && fifo_empty == 1 && fifo_tile_valid == 0) begin
                tile_x_loc         <= subtile_x_loc[0];
                tile_y_loc         <= subtile_y_loc[0];
                level              <= level_1;
                tile_edge_func_val <= subtile_edge_func_val[0];
            end
            // else ready from the fifo is it is not empty and fifo tile is valid
            else if (fifo_empty == 0 && fifo_tile_valid == 1) begin
                tile_x_loc         <= fifo_tile_x_loc;
                tile_y_loc         <= fifo_tile_y_loc;
                level              <= fifo_tile_level;
                tile_edge_func_val <= fifo_tile_edge_func_val;
            end
        end
    end

    // Decide the ready flag
    //  1. Tile evaluator doesn't have a valid tile or (block -> block will be pushed to next pipe so no need to stall for it)
    //  2. FIFO empty
    //  3. FIFO pop data is invalid
    assign ready = (fifo_empty == 1) && (block_fifo_empty == 1) && (valid_tile == 0);

    // Sub-tile data output from tile-evaluator
    logic [`RASTER_TILE_DATA_BITS-1:0]      subtile_x_loc[3:0], subtile_y_loc[3:0];
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] subtile_edge_func_val[3:0][2:0];

    /**********************************
            TILE EVALUATOR
    ***********************************/
    VX_raster_te #(
        .RASTER_TILE_SIZE(RASTER_TILE_SIZE),
        .RASTER_BLOCK_SIZE(RASTER_BLOCK_SIZE),
        .RASTER_LEVEL_DATA_BITS(RASTER_LEVEL_DATA_BITS)
    ) tile_evaluator (
        .level(level),
        .x_loc(tile_x_loc),
        .y_loc(tile_y_loc),
        .edges(global_edges),
        .edge_func_val(tile_edge_func_val),
        .extents(global_extents),
        .valid_tile(valid_tile),
        .valid_block(valid_block),
        .tile_x_loc(subtile_x_loc),
        .tile_y_loc(subtile_y_loc),
        .tile_edge_func_val(subtile_edge_func_val)
    );

    /**********************************
            TILE ARBITER
    ***********************************/

   // Create mask for sub-tile push into fifo
    logic [3:0] fifo_push_mask;
    for (genvar i = 0; i < 4; ++i) begin
        assign fifo_push_mask[i] = ~(i == 0 && level == 0);
    end

    // Create data_push data
    logic [RASTER_FIFO_DATA_WIDTH-1:0] fifo_data_push[3:0];
    for (genvar i = 0; i < 4; ++i) begin
        assign fifo_data_push[i] = {level_1, subtile_x_loc[i], subtile_y_loc[i], 
            subtile_edge_func_val[i][0], subtile_edge_func_val[i][1],
            subtile_edge_func_val[i][2]};
    end

    logic [RASTER_FIFO_DATA_WIDTH-1:0] fifo_data_pop;
    // Create sub-tile data from the fifo output
    assign {fifo_tile_level, fifo_tile_x_loc, fifo_tile_y_loc, fifo_tile_edge_func_val[0],
        fifo_tile_edge_func_val[1], fifo_tile_edge_func_val[2]} = fifo_data_pop;

    // Assert that fifo cannot be full when tile is valid
    always_comb
        `ASSERT(!(fifo_full == 1 && valid_tile == 1), ("Raster insufficient subtile fifo depth"));
    // NOTE: condition not added in fifo_push check as it wil lead to deadlock => Assertion added

    // Set the pop logic based on stall if not stalled & not empty, then it will definitely pop
    logic [3:0] fifo_pop, fifo_index_onehot;
    for (genvar i = 0; i < 4; ++i) begin
        // Updated based on conditions and from the onehot received from arbiter
        assign fifo_pop[i] = (stall == 0) && (fifo_empty == 0) && (fifo_index_onehot[i] == 1);
    end

    VX_raster_te_arbiter #(
        .RASTER_TILE_SIZE(RASTER_TILE_SIZE),
        .RASTER_BLOCK_SIZE(RASTER_BLOCK_SIZE)
    ) tile_arbiter (
        .clk(clk), 
        .reset(reset),
        .fifo_push({4{valid_tile}} & fifo_push_mask),  // Push only tiles, not blocks
        .fifo_pop(fifo_pop),
        .data_push(fifo_data_push),
        .data_pop(fifo_data_pop),
        .fifo_full(fifo_full),
        .fifo_empty(fifo_empty),
        .fifo_data_valid(fifo_tile_valid),
        .fifo_index_onehot(fifo_index_onehot)
    );


    /**********************************
            BLOCK EVALUATOR
    ***********************************/

    logic block_fifo_full, block_fifo_empty;

    // Block evaluator data
    logic [`RASTER_TILE_DATA_BITS-1:0]   be_in_x_loc, be_in_y_loc;
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0] be_in_edge_func_val[2:0];

    localparam BLOCK_FIFO_DATA_WIDTH = 2*`RASTER_TILE_DATA_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS;

    logic be_fifo_pop;
    assign be_fifo_pop = (be_ready == 1 && block_fifo_empty == 0);

    // Stop pushing the last tile back in
    //logic last_block;
    always @(posedge clk) begin
        // check if the current block is going to be the last block
        if (fifo_empty == 1 && valid_tile == 0 && valid_block == 1)
            done <= 1;
    end

    // Block fifo
    VX_fifo_queue #(
        .DATAW	    (BLOCK_FIFO_DATA_WIDTH),
        .SIZE       (RASTER_TILE_FIFO_DEPTH),
        .OUT_REG    (1)
    ) block_fifo_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (valid_block == 1 && block_fifo_full == 0 && done == 0),
        .pop        (be_fifo_pop),
        .data_in    ({
            tile_x_loc, tile_y_loc,
            tile_edge_func_val[0], tile_edge_func_val[1], tile_edge_func_val[2]
        }),
        .data_out   ({
            be_in_x_loc, be_in_y_loc,
            be_in_edge_func_val[0], be_in_edge_func_val[1], be_in_edge_func_val[2]
        }),
        .full       (block_fifo_full),
        .empty      (block_fifo_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    VX_raster_be #(
        .RASTER_BLOCK_SIZE(RASTER_BLOCK_SIZE),
        .RASTER_QUAD_OUTPUT_RATE(RASTER_QUAD_OUTPUT_RATE),
        .RASTER_QUAD_FIFO_DEPTH(RASTER_QUAD_FIFO_DEPTH)
    ) block_evaluator (
        .clk(clk),
        .reset(reset),
        .input_valid(be_fifo_pop),
        .x_loc(be_in_x_loc),
        .y_loc(be_in_y_loc),
        .edges(global_edges),
        .edge_func_val(be_in_edge_func_val),
        .out_quad_x_loc(out_quad_x_loc),
        .out_quad_y_loc(out_quad_y_loc),
        .out_quad_masks(out_quad_masks),
        .valid(valid), 
        .ready(be_ready),
        .pop(pop_quad),
        .empty(quad_queue_empty)
    );

endmodule
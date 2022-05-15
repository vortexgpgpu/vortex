// Raster slice
// Functionality:
//     1. Recursive descent
//     2. Tile evaluation
//     3. Quad evaluation and storage
//     4. Return the quad(s)

`include "VX_raster_define.vh"

module VX_raster_slice #(
    parameter CLUSTER_ID       = 0,
    parameter SLICE_ID         = 1,
    parameter BLOCK_SIZE       = 8,
    parameter TILE_SIZE        = 16,
    parameter OUTPUT_QUADS     = 4,
    parameter QUAD_FIFO_DEPTH  = 1,
    parameter TILE_FIFO_DEPTH  = 16
) (
    // Standard inputs
    input wire                                             clk, reset,
    // To indicate valid input provided
    input wire                                             input_valid,
    // Tile information
    input wire        [`RASTER_DIM_BITS-1:0]               x_loc, y_loc,
    // Primitive information
    input wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    edges[2:0][2:0],
    input wire        [`RASTER_PRIMITIVE_DATA_BITS-1:0]    pid,
    input wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    edge_func_val[2:0],
    input wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    extents[2:0],
    // Rendering region
    input wire         [`RASTER_DIM_BITS-1:0]              dst_width, dst_height,

    // Hand-shaking signals
    input wire                                             pop_quad,
    output wire                                            ready, quad_queue_empty,

    // Output sub-tiles data
    output wire        [`RASTER_DIM_BITS-1:0]              out_quad_x_loc  [OUTPUT_QUADS-1:0],
                                                            out_quad_y_loc  [OUTPUT_QUADS-1:0],
    output wire        [3:0]                               out_quad_masks  [OUTPUT_QUADS-1:0],
    output wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   out_quad_bcoords[OUTPUT_QUADS-1:0][2:0][3:0],
    output wire        [`RASTER_PRIMITIVE_DATA_BITS-1:0]   out_pid         [OUTPUT_QUADS-1:0],
    output wire        [OUTPUT_QUADS-1:0]                  valid
);

    localparam RASTER_LEVEL_DATA_BITS = $clog2(TILE_SIZE/BLOCK_SIZE) + 1;
    localparam RASTER_FIFO_DATA_WIDTH = (RASTER_LEVEL_DATA_BITS + 2*`RASTER_DIM_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS);

    // Store data which will stay same for tile throughout operation
    reg signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  global_extents[2:0];
    reg signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  global_edges[2:0][2:0];
    reg        [`RASTER_PRIMITIVE_DATA_BITS-1:0]  global_pid;

    // Store the tile relevant data as global regs as TE is combinatorial
    reg        [`RASTER_DIM_BITS-1:0]             tile_x_loc, tile_y_loc;
    reg        [RASTER_LEVEL_DATA_BITS-1:0]       level;
    wire       [RASTER_LEVEL_DATA_BITS-1:0]       level_r;
    reg signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  tile_edge_func_val[2:0];

    // Control signsl
    wire        tile_valid, block_valid;
    wire        fifo_full, fifo_empty, fifo_tile_valid;
    wire        stall;
    wire        be_ready; // to track the status of the block evaluator
    reg        done;

    // Incoming tile data from fifo
    wire        [`RASTER_DIM_BITS-1:0]             fifo_tile_x_loc, fifo_tile_y_loc;
    wire        [RASTER_LEVEL_DATA_BITS-1:0]       fifo_tile_level;
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  fifo_tile_edge_func_val[2:0];

    // Sub-tile data output from tile-evaluator
    wire        [`RASTER_DIM_BITS-1:0]             subtile_x_loc[3:0], subtile_y_loc[3:0];
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  subtile_edge_func_val[3:0][2:0];

    // Block data output from the tile-evaluator
    wire        [`RASTER_DIM_BITS-1:0]             block_x_loc, block_y_loc;
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  block_edge_func_val[2:0];

    reg tile_data_valid;

    // Tile data selector to choose tile data from:
    //     1. Input tile
    //     2. Sub-tile for queue
    //     3. Data forwarded sub-tile for first iteration when
    //        input tile divided into 4, but sub-tile fifo will
    //        be empty. So, instead of wasting 1 cycle, sub=tile
    //        forwarded directly here.
    always @(posedge clk) begin
        tile_data_valid <= 0;
        if (reset) begin
            // Reset all globals and signals
            for (integer i = 0; i < 3; ++i) begin
                global_extents[i]      <= 0;
                tile_edge_func_val[i]  <= 0;
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
                global_pid         <= pid;
                tile_data_valid    <= 1;
            end
            // sub-tile rerouter used only 1 onces for the parent tile
            else if (level == 0 && fifo_empty == 1 && fifo_tile_valid == 0 && tile_valid == 1) begin
                tile_x_loc         <= subtile_x_loc[0];
                tile_y_loc         <= subtile_y_loc[0];
                level              <= level + RASTER_LEVEL_DATA_BITS'(1);//level_1;
                tile_edge_func_val <= subtile_edge_func_val[0];
                tile_data_valid    <= 1;
            end
            // else ready from the fifo is it is not empty and fifo tile is valid
            else if (fifo_empty == 0 && fifo_tile_valid == 1) begin
                tile_x_loc         <= fifo_tile_x_loc;
                tile_y_loc         <= fifo_tile_y_loc;
                level              <= fifo_tile_level;
                tile_edge_func_val <= fifo_tile_edge_func_val;
                tile_data_valid    <= 1;
            end
        end
    end

    /**********************************
            TILE EVALUATOR
    ***********************************/

    wire input_valid_r;
    wire        [RASTER_LEVEL_DATA_BITS-1:0]         level_te_pipe_r;
    wire        [`RASTER_DIM_BITS-1:0]               x_loc_r, y_loc_r;
    wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]    edge_func_val_r[2:0];

    VX_pipe_register #(
        .DATAW  (1 + RASTER_LEVEL_DATA_BITS + 2*`RASTER_DIM_BITS
            + 3*`RASTER_PRIMITIVE_DATA_BITS),
        .RESETW (1)
    ) te_pipe_reg_1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({
            tile_data_valid, level, tile_x_loc, tile_y_loc,
            tile_edge_func_val[0], tile_edge_func_val[1], tile_edge_func_val[2]
        }),
        .data_out ({
            input_valid_r, level_te_pipe_r, x_loc_r, y_loc_r,
            edge_func_val_r[0], edge_func_val_r[1], edge_func_val_r[2]
        })
    );

    VX_raster_te #(
        .TILE_SIZE       (TILE_SIZE),
        .BLOCK_SIZE      (BLOCK_SIZE),
        .RASTER_LEVEL_DATA_BITS (RASTER_LEVEL_DATA_BITS)
    ) tile_evaluator (
        .clk                    (clk),
        .reset                  (reset),
        .stall                  (stall),
        .input_valid            (input_valid_r),
        .level                  (level_te_pipe_r),
        .x_loc                  (x_loc_r),
        .y_loc                  (y_loc_r),
        .edges                  (global_edges),
        .edge_func_val          (edge_func_val_r),
        .extents                (global_extents),
        .tile_valid             (tile_valid),
        .block_valid            (block_valid),
        .tile_x_loc             (subtile_x_loc),
        .tile_y_loc             (subtile_y_loc),
        .tile_edge_func_val     (subtile_edge_func_val),
        .block_x_loc            (block_x_loc),
        .block_y_loc            (block_y_loc),
        .block_edge_func_val    (block_edge_func_val),
        .tile_level             (level_r)
    );

    /**********************************
            TILE ARBITER
    ***********************************/

   // Create mask for sub-tile push into fifo
    wire [3:0] fifo_push_mask;
    for (genvar i = 0; i < 4; ++i) begin
        assign fifo_push_mask[i] = ~(i == 0 && level_r == 0);
    end

    // Create data_push data
    wire [RASTER_FIFO_DATA_WIDTH-1:0] fifo_data_push[3:0];
    for (genvar i = 0; i < 4; ++i) begin
        assign fifo_data_push[i] = {level_r + RASTER_LEVEL_DATA_BITS'(1), subtile_x_loc[i], subtile_y_loc[i], 
            subtile_edge_func_val[i][0],
            subtile_edge_func_val[i][1],
            subtile_edge_func_val[i][2]};
    end

    wire [RASTER_FIFO_DATA_WIDTH-1:0] fifo_data_pop;
    // Create sub-tile data from the fifo output
    assign {fifo_tile_level, fifo_tile_x_loc, fifo_tile_y_loc, fifo_tile_edge_func_val[0],
        fifo_tile_edge_func_val[1], fifo_tile_edge_func_val[2]} = fifo_data_pop;

    // Assert that fifo cannot be full when tile is valid
    always_comb
        `ASSERT(!(fifo_full == 1 && tile_valid == 1), ("Raster insufficient subtile fifo depth"));
    // NOTE: condition not added in fifo_push check as it wil lead to deadlock => Assertion added

    // Set the pop logic based on stall if not stalled & not empty, then it will definitely pop
    wire [3:0] fifo_pop, fifo_index_onehot;
    for (genvar i = 0; i < 4; ++i) begin
        // Updated based on conditions and from the onehot received from arbiter
        assign fifo_pop[i] = (stall == 0) && (fifo_empty == 0) && (fifo_index_onehot[i] == 1);
    end

    VX_raster_te_arbiter #(
        .TILE_SIZE       (TILE_SIZE),
        .BLOCK_SIZE      (BLOCK_SIZE),
        .TILE_FIFO_DEPTH (TILE_FIFO_DEPTH)
    ) tile_arbiter (
        .clk                (clk),
        .reset              (reset),
        .fifo_push          ({4{tile_valid}} & fifo_push_mask),  // Push only tiles, not blocks
        .fifo_pop           (fifo_pop),
        .data_push          (fifo_data_push),
        .data_pop           (fifo_data_pop),
        .fifo_full          (fifo_full),
        .fifo_empty         (fifo_empty),
        .fifo_data_valid    (fifo_tile_valid),
        .fifo_index_onehot  (fifo_index_onehot)
    );


    /**********************************
            BLOCK EVALUATOR
    ***********************************/

    wire block_fifo_full, block_fifo_empty;

    
    // Stall used to wait for block queue to complete run if another needs to be inserted
    //assign stall = (block_valid == 1 && block_fifo_full == 1);
    assign stall = (block_fifo_full == 1);

    // Decide the ready flag
    //  1. Tile evaluator doesn't have a valid tile or (block -> block will be pushed to next pipe so no need to stall for it)
    //  2. FIFO empty
    //  3. FIFO pop data is invalid
    assign ready = (fifo_empty == 1) && (block_fifo_empty == 1) && (tile_valid == 0) && (block_valid == 0) &&
        (tile_data_valid == 0) && (input_valid_r == 0);

    // Block evaluator data
    wire [`RASTER_DIM_BITS-1:0]   be_in_x_loc, be_in_y_loc;
    wire [`RASTER_PRIMITIVE_DATA_BITS-1:0] be_in_edge_func_val[2:0];

    localparam BLOCK_FIFO_DATA_WIDTH = 2*`RASTER_DIM_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS;

    wire be_fifo_pop;
    assign be_fifo_pop = (be_ready == 1 && block_fifo_empty == 0);

    // Stop pushing the last tile back in
    //logic last_block;
    always @(posedge clk) begin
        done <= 0;
        // check if the current block is going to be the last block
        if (fifo_empty == 1 && tile_valid == 0 && block_valid == 1 && tile_data_valid == 0 && input_valid_r  == 0)
            done <= 1;
    end

    // Block fifo
    VX_fifo_queue #(
        .DATAW	    (BLOCK_FIFO_DATA_WIDTH),
        .SIZE       (TILE_FIFO_DEPTH),
        .OUT_REG    (1)
    ) block_fifo_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (block_valid == 1 && block_fifo_full == 0 && done == 0),
        .pop        (be_fifo_pop),
        .data_in    ({
            block_x_loc, block_y_loc,
            block_edge_func_val[0], block_edge_func_val[1], block_edge_func_val[2]
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
        .BLOCK_SIZE      (BLOCK_SIZE),
        .OUTPUT_QUADS(OUTPUT_QUADS),
        .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH),
        .SLICE_ID               (SLICE_ID)
    ) block_evaluator (
        .clk                    (clk),
        .reset                  (reset),
        .input_valid            (be_fifo_pop),
        .x_loc                  (be_in_x_loc),
        .y_loc                  (be_in_y_loc),
        .edges                  (global_edges),
        .pid                    (global_pid),
        .edge_func_val          (be_in_edge_func_val),
        .out_quad_x_loc         (out_quad_x_loc),
        .out_quad_y_loc         (out_quad_y_loc),
        .out_quad_masks         (out_quad_masks),
        .out_quad_bcoords       (out_quad_bcoords),
        .out_pid                (out_pid),
        .valid                  (valid), 
        .ready                  (be_ready),
        .pop                    (pop_quad),
        .empty                  (quad_queue_empty),
        .dst_width              (dst_width),
        .dst_height             (dst_height)
    );

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (input_valid) begin
            dpi_trace(2, "%d: raster-slice-in[%0d]: x=%0d, y=%0d, pid=%0d, edge1.x=%0d, edge1.y=%0d, edge1.z=%0d, edge2.x=%0d, edge2.y=%0d, edge2.z=%0d, edge3.x=%0d, edge3.y=%0d, edge3.z=%0d, edge_func_val=%0d %0d %0d, extents=%0d %0d %0d\n",
                $time, SLICE_ID, x_loc, y_loc, pid,
                edges[0][0], edges[0][1], edges[0][2],
                edges[1][0], edges[1][1], edges[1][2],
                edges[2][0], edges[2][1], edges[2][2],
                edge_func_val[0], edge_func_val[1], edge_func_val[2],
                extents[0], extents[1], extents[2]);
        end
    end

    always @(posedge clk) begin
        if (block_valid == 1 && block_fifo_full == 0 && done == 0) begin
            dpi_trace(2, "%d: block-fifo-push[%0d]: data_in=%0d, full=%b, empty=%b\n",
            $time, SLICE_ID, {block_x_loc, block_y_loc,
                block_edge_func_val[0], block_edge_func_val[1], block_edge_func_val[2]},
                block_fifo_full, block_fifo_empty);
        end
    end

`endif

endmodule

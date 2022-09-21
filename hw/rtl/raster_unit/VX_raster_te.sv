// Tile evaluator
// Functionality: Receives a tile
//     1. Recursive descend sub-tiles that overlap primitive
//     2. Stop when tile size matches block

`include "VX_raster_define.vh"

module VX_raster_te #(
    parameter `STRING_TYPE INSTANCE_ID = "",
    parameter TILE_LOGSIZE  = 5,
    parameter BLOCK_LOGSIZE = 2  
) (
    input wire clk,
    input wire reset,

    output wire                         empty,

    // Inputs
    input wire                          valid_in,
    input wire [`RASTER_DIM_BITS-1:0]   x_loc_in,
    input wire [`RASTER_DIM_BITS-1:0]   y_loc_in,
    input wire [`RASTER_PID_BITS-1:0]   pid_in,
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_in,
    input wire [2:0][`RASTER_DATA_BITS-1:0] extents_in,
    output wire                         ready_in,
    
    // Outputs
    output wire                         valid_out,
    output wire [`RASTER_DIM_BITS-1:0]  x_loc_out,
    output wire [`RASTER_DIM_BITS-1:0]  y_loc_out,
    output wire [`RASTER_PID_BITS-1:0]  pid_out,
    output wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_out,
    input wire                          ready_out
);
    localparam LEVEL_BITS      = (TILE_LOGSIZE - BLOCK_LOGSIZE) + 1;
    localparam TILE_FIFO_DEPTH = 1 << (2 * (TILE_LOGSIZE - BLOCK_LOGSIZE));
    localparam FIFO_DATA_WIDTH = 2 * `RASTER_DIM_BITS + 3 * `RASTER_DATA_BITS + LEVEL_BITS;

    wire stall;

    reg [2:0][`RASTER_DATA_BITS-1:0] tile_extents;
    reg [2:0][2:0][`RASTER_DATA_BITS-1:0] tile_edges;
    reg [`RASTER_PID_BITS-1:0]       tile_pid;
    reg [`RASTER_DIM_BITS-1:0]       tile_x_loc;
    reg [`RASTER_DIM_BITS-1:0]       tile_y_loc;
    reg [2:0][`RASTER_DATA_BITS-1:0] tile_edge_eval;
    reg [LEVEL_BITS-1:0]             tile_level;

    wire [`RASTER_DIM_BITS-1:0]       tile_x_loc_r;
    wire [`RASTER_DIM_BITS-1:0]       tile_y_loc_r;
    wire [2:0][`RASTER_DATA_BITS-1:0] tile_edge_eval_r;
    wire [LEVEL_BITS-1:0]             tile_level_r;

    wire [3:0][`RASTER_DIM_BITS-1:0] subtile_x_loc, subtile_x_loc_r;
    wire [3:0][`RASTER_DIM_BITS-1:0] subtile_y_loc, subtile_y_loc_r;
    wire [3:0][2:0][`RASTER_DATA_BITS-1:0] subtile_edge_eval, subtile_edge_eval_r;
    wire [LEVEL_BITS-1:0] subtile_level, subtile_level_r;

    wire [`RASTER_DIM_BITS-1:0] fifo_x_loc;
    wire [`RASTER_DIM_BITS-1:0] fifo_y_loc;
    wire [2:0][`RASTER_DATA_BITS-1:0] fifo_edge_eval;
    wire [LEVEL_BITS-1:0]  fifo_level;

    wire       fifo_arb_valid;    
    wire [1:0] fifo_arb_index;
    wire [3:0] fifo_arb_onehot;

    reg  tile_valid;
    wire tile_valid_r;
    wire is_block_r;

    // fifo bypass first sub-tile
    wire is_fifo_bypass = tile_valid_r && ~is_block_r && ~fifo_arb_valid;

    always @(posedge clk) begin
        if (reset) begin
            tile_valid <= 0;
        end else begin
            if (~stall) begin
                tile_valid <= 0;
                if (fifo_arb_valid) begin
                    // select fifo input
                    tile_valid          <= 1;
                    tile_x_loc          <= fifo_x_loc;
                    tile_y_loc          <= fifo_y_loc;                
                    tile_edge_eval      <= fifo_edge_eval;                
                    tile_level          <= fifo_level;
                end else 
                if (is_fifo_bypass) begin
                    // fifo bypass first sub-tile
                    tile_valid          <= 1;
                    tile_x_loc          <= subtile_x_loc_r[0];
                    tile_y_loc          <= subtile_y_loc_r[0];                
                    tile_edge_eval      <= subtile_edge_eval_r[0];                
                    tile_level          <= subtile_level_r;
                end else
                if (valid_in && ~tile_valid) begin   
                    // select new tile input             
                    tile_valid          <= 1;
                    tile_extents        <= extents_in;
                    tile_edges          <= edges_in;
                    tile_pid            <= pid_in;         
                    tile_x_loc          <= x_loc_in;
                    tile_y_loc          <= y_loc_in;
                    tile_edge_eval[0]   <= edges_in[0][2];
                    tile_edge_eval[1]   <= edges_in[1][2];
                    tile_edge_eval[2]   <= edges_in[2][2];
                    tile_level          <= 0;
                end
            end
        end
    end

    // Generate sub-tile info
    wire [`RASTER_DIM_BITS-1:0] tile_logsize = `RASTER_DIM_BITS'(TILE_LOGSIZE-1) - `RASTER_DIM_BITS'(tile_level);
    wire is_block = (tile_logsize < `RASTER_DIM_BITS'(BLOCK_LOGSIZE));
    assign subtile_level = tile_level + LEVEL_BITS'(1);
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            assign subtile_x_loc[2 * i + j] = tile_x_loc + (`RASTER_DIM_BITS'(i) << tile_logsize);
            assign subtile_y_loc[2 * i + j] = tile_y_loc + (`RASTER_DIM_BITS'(j) << tile_logsize);
            for (genvar k = 0; k < 3; ++k) begin
                assign subtile_edge_eval[2 * i + j][k] = i * (tile_edges[k][0] << tile_logsize) + j * (tile_edges[k][1] << tile_logsize) + tile_edge_eval[k];
            end
        end
    end

    // Check if primitive overlaps current tile
    wire [2:0][`RASTER_DATA_BITS-1:0] edge_eval;
    for (genvar i = 0; i < 3; ++i) begin
        assign edge_eval[i] = tile_edge_eval[i] + (tile_extents[i] >> tile_level);
    end
    wire overlap = ~(edge_eval[0][`RASTER_DATA_BITS-1] 
                  || edge_eval[1][`RASTER_DATA_BITS-1] 
                  || edge_eval[2][`RASTER_DATA_BITS-1]);

    wire tile_valid_e = tile_valid && overlap;

    VX_pipe_register #(
        .DATAW  (1 + 1 + 4 * (2 * `RASTER_DIM_BITS + 3 * `RASTER_DATA_BITS) + LEVEL_BITS + 2 * `RASTER_DIM_BITS + 3 * `RASTER_DATA_BITS + LEVEL_BITS),
        .RESETW (1)
    ) te_pipe_reg_1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({tile_valid_e, is_block,   subtile_x_loc,   subtile_y_loc,   subtile_edge_eval,   subtile_level,   tile_x_loc,   tile_y_loc,   tile_edge_eval,   tile_level}),
        .data_out ({tile_valid_r, is_block_r, subtile_x_loc_r, subtile_y_loc_r, subtile_edge_eval_r, subtile_level_r, tile_x_loc_r, tile_y_loc_r, tile_edge_eval_r, tile_level_r})
    );

    wire [3:0] fifo_full, fifo_empty;
    wire [3:0][FIFO_DATA_WIDTH-1:0] fifo_data_out;

    wire fifo_stall   = tile_valid_r && ~is_block_r && (| fifo_full);
    wire output_stall = tile_valid_r && is_block_r && ~ready_out;

    for (genvar i = 0; i < 4; ++i) begin
        wire fifo_push = tile_valid_r && ~is_block_r && ~(is_fifo_bypass && i == 0);
        wire fifo_pop = ~fifo_empty[i] && fifo_arb_onehot[i] && ~output_stall; 

        VX_fifo_queue #(
            .DATAW	 (FIFO_DATA_WIDTH),
            .SIZE    (TILE_FIFO_DEPTH),
            .OUT_REG (1)
        ) fifo_queue (
            .clk        (clk),
            .reset      (reset),
            .push       (fifo_push),
            .pop        (fifo_pop),
            .data_in    ({subtile_x_loc_r[i], subtile_y_loc_r[i], subtile_edge_eval_r[i], subtile_level_r}),
            .data_out   (fifo_data_out[i]),
            .full       (fifo_full[i]),
            .empty      (fifo_empty[i]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (size)
        );
    end

    assign {fifo_x_loc, fifo_y_loc, fifo_edge_eval, fifo_level} = fifo_data_out[fifo_arb_index];

    VX_priority_arbiter #(
        .NUM_REQS (4)
    ) fifo_arbiter (
        .clk          (clk),
        .reset        (reset),        
        `UNUSED_PIN   (unlock),
        .requests     (~fifo_empty),
        .grant_index  (fifo_arb_index),
        .grant_onehot (fifo_arb_onehot),
        .grant_valid  (fifo_arb_valid)
    );

    // pipeline stall
    assign stall = fifo_stall || output_stall;

    // can accept next input?
    assign ready_in = ~stall           // no pipeline stall
                   && ~tile_valid      // no tile in process
                   && ~fifo_arb_valid  // no fifo input
                   && ~is_fifo_bypass; // no fifo bypass

    assign valid_out = tile_valid_r && is_block_r;
    assign x_loc_out = tile_x_loc_r;
    assign y_loc_out = tile_y_loc_r;
    assign pid_out   = tile_pid;    
    `EDGE_UPDATE (edges_out, tile_edges, tile_edge_eval_r);

    assign empty = ready_in && ~valid_out;

    `UNUSED_VAR (tile_level_r)

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (valid_in && ready_in) begin
            `TRACE(2, ("%d: %s-te-in: x=%0d, y=%0d, edge={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}, extents={0x%0h, 0x%0h, 0x%0h}\n",
                $time, INSTANCE_ID, x_loc_in, y_loc_in,
                edges_in[0][0], edges_in[0][1], edges_in[0][2],
                edges_in[1][0], edges_in[1][1], edges_in[1][2],
                edges_in[2][0], edges_in[2][1], edges_in[2][2],
                extents_in[0],  extents_in[1],  extents_in[2]));
        end
        if (tile_valid && ~stall) begin
            `TRACE(2, ("%d: %s-te-test: pass=%b, block=%b, level=%0d, x=%0d, y=%0d, edge_eval={0x%0h, 0x%0h, 0x%0h}\n",
                $time, INSTANCE_ID, tile_valid_e, is_block, tile_level, tile_x_loc, tile_y_loc, edge_eval[0], edge_eval[1], edge_eval[2]));
        end
    end
`endif

endmodule

`include "VX_raster_define.vh"

module VX_raster_pe #(
    parameter `STRING_TYPE INSTANCE_ID = "",
    parameter TILE_LOGSIZE    = 5,
    parameter BLOCK_LOGSIZE   = 2,
    parameter OUTPUT_QUADS    = 4,
    parameter QUAD_FIFO_DEPTH = 4
) (
    input wire clk,
    input wire reset,

    // Device configurations
    raster_dcrs_t                                   dcrs,

    // Inputs
    input wire                                      valid_in,
    input wire [`RASTER_DIM_BITS-1:0]               x_loc_in,
    input wire [`RASTER_DIM_BITS-1:0]               y_loc_in,
    input wire [`RASTER_DIM_BITS-1:0]               x_max_in,
    input wire [`RASTER_DIM_BITS-1:0]               y_min_in,
    input wire [`RASTER_DIM_BITS-1:0]               y_max_in,
    input wire [`RASTER_PID_BITS-1:0]               pid_in,
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0]    edges_in,    
    input wire [2:0][`RASTER_DATA_BITS-1:0]         extents_in,    
    output wire                                     ready_in, 

    // Outputs
    output wire                                     valid_out,
    output raster_stamp_t [OUTPUT_QUADS-1:0]        stamps_out, 
    output wire                                     empty_out,
    input  wire                                     ready_out
);
    localparam NUM_QUADS_DIM   = 1 << (BLOCK_LOGSIZE - 1);
    localparam PER_BLOCK_QUADS = NUM_QUADS_DIM * NUM_QUADS_DIM;
    localparam OUTPUT_BATCHES  = (PER_BLOCK_QUADS + OUTPUT_QUADS - 1) / OUTPUT_QUADS;
    localparam BLOCK_BUF_SIZE  = 2 * OUTPUT_BATCHES;

    wire te_empty;
    wire be_empty;

    wire                        block_valid;
    wire [`RASTER_DIM_BITS-1:0] block_x_loc;
    wire [`RASTER_DIM_BITS-1:0] block_y_loc;
    wire [`RASTER_PID_BITS-1:0] block_pid;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] block_edges;
    wire                        block_ready;
    
    VX_raster_te #(
        .INSTANCE_ID   (INSTANCE_ID),
        .TILE_LOGSIZE  (TILE_LOGSIZE),
        .BLOCK_LOGSIZE (BLOCK_LOGSIZE)
    ) tile_evaluator (
        .clk        (clk),
        .reset      (reset),

        .empty      (te_empty),
        
        .valid_in   (valid_in),
        .x_loc_in   (x_loc_in),
        .y_loc_in   (y_loc_in),
        .pid_in     (pid_in),
        .edges_in   (edges_in),
        .extents_in (extents_in),
        .ready_in   (ready_in),

        .valid_out  (block_valid),
        .x_loc_out  (block_x_loc),
        .y_loc_out  (block_y_loc),
        .pid_out    (block_pid),
        .edges_out  (block_edges),  
        .ready_out  (block_ready)
    );

    wire                        block_valid_b;
    wire [`RASTER_DIM_BITS-1:0] block_x_loc_b;
    wire [`RASTER_DIM_BITS-1:0] block_y_loc_b;
    wire [`RASTER_PID_BITS-1:0] block_pid_b;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] block_edges_b;
    wire                        block_ready_b;

    VX_elastic_buffer #(
        .DATAW (2 * `RASTER_DIM_BITS + `RASTER_PID_BITS + 9 * `RASTER_DATA_BITS),
        .SIZE  (BLOCK_BUF_SIZE)
    ) block_req_buf (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (block_valid),
        .ready_in   (block_ready),
        .data_in    ({block_x_loc,   block_y_loc,   block_pid,   block_edges}),
        .data_out   ({block_x_loc_b, block_y_loc_b, block_pid_b, block_edges_b}),
        .valid_out  (block_valid_b),
        .ready_out  (block_ready_b)
    );

    VX_raster_be #(
        .INSTANCE_ID     (INSTANCE_ID),
        .BLOCK_LOGSIZE   (BLOCK_LOGSIZE),
        .OUTPUT_QUADS    (OUTPUT_QUADS),
        .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH)
    ) block_evaluator (
        .clk        (clk),
        .reset      (reset),
        
        .dcrs       (dcrs),

        .empty      (be_empty),

        .valid_in   (block_valid_b),
        .x_loc_in   (block_x_loc_b),
        .y_loc_in   (block_y_loc_b),
        .x_max_in   (x_max_in),
        .y_min_in   (y_min_in),
        .y_max_in   (y_max_in),
        .pid_in     (block_pid_b),
        .edges_in   (block_edges_b),
        .ready_in   (block_ready_b),
        
        .valid_out  (valid_out),
        .stamps_out (stamps_out),
        .ready_out  (ready_out)
    );

    assign empty_out = te_empty
                    && ~block_valid_b 
                    && be_empty;

endmodule

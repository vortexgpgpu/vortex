
// Rasterization tile arbiter
// Functionality: Stores 4 sub-tiles and returns 1 based on arbitration

`include "VX_raster_define.vh"

module VX_raster_te_arbiter #(
    parameter TILE_SIZE       = 16,
    parameter BLOCK_SIZE      = 4,
    parameter TILE_FIFO_DEPTH = 32,
    parameter LEVEL_DATA_BITS = $clog2(TILE_SIZE / BLOCK_SIZE) + 1,
    parameter FIFO_DATA_WIDTH = (LEVEL_DATA_BITS + 2 * `RASTER_DIM_BITS + 3 * `RASTER_PRIMITIVE_DATA_BITS)
) (
    input wire                        clk,
    input wire                        reset,

    input wire [3:0]                  fifo_push,
    input wire [3:0]                  fifo_pop,
    input wire [FIFO_DATA_WIDTH-1:0]  data_push[3:0],

    output wire [FIFO_DATA_WIDTH-1:0] data_pop,    
    output wire [3:0]                 fifo_index_onehot,
    output wire                       fifo_data_valid,
    output wire                       fifo_full,
    output wire                       fifo_empty
);

    // localparam TILE_FIFO_DEPTH = (TILE_SIZE*TILE_SIZE)/(BLOCK_SIZE*BLOCK_SIZE);

    // Per FIFO flags
    wire [3:0] empty_flag, full_flag;
    wire [FIFO_DATA_WIDTH-1:0] data_pop_array[3:0];

    // Index selected from arbitration
    wire [1:0] fifo_index;

    // Generate 4 queues for 4 sub-tiles
    for(genvar i = 0; i < 4; ++i) begin
        // Sub-tile queue
        VX_fifo_queue #(
            .DATAW	    (FIFO_DATA_WIDTH),
            .SIZE       (TILE_FIFO_DEPTH),
            .OUT_REG    (1)
        ) tile_fifo_queue (
            .clk        (clk),
            .reset      (reset),
            .push       (fifo_push[i]),
            .pop        (fifo_pop[i]),
            .data_in    (data_push[i]),
            .data_out   (data_pop_array[i]),
            .full       (full_flag[i]),
            .empty      (empty_flag[i]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (size)
        );
    end

    assign fifo_empty = &empty_flag;
    assign fifo_full =  &full_flag;

    // Arbitrate over the available entries to pop and generate index to pop for sub=tile
    VX_rr_arbiter #(
        .NUM_REQS   (4)
    ) tile_fifo_arbiter (
        .clk            (clk),
        .reset          (reset),
        `UNUSED_PIN     (unlock),
        .requests       (~empty_flag),
        .grant_index    (fifo_index),
        .grant_onehot   (fifo_index_onehot),
        .grant_valid    (fifo_data_valid)
    );

    assign data_pop = data_pop_array[fifo_index];

endmodule

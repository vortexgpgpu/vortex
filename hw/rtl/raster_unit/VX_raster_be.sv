// Block level evaluator
// Functionality: Receives a block of AxA (where A is pow(2))
//     1. Breaks it into quad and runs quad evaluators on it
//     2. Stores the result in quad queues
//     3. Queues direction read as outputs

`include "VX_raster_define.vh"

module VX_raster_be #(
    parameter SLICE_ID        = 0,
    parameter BLOCK_LOGSIZE   = 5,
    parameter OUTPUT_QUADS    = 2,
    parameter QUAD_FIFO_DEPTH = 4    
) (
    // Standard inputs
    input wire clk,
    input wire reset,

    // Device configurations
    raster_dcrs_t dcrs,

    output wire                         empty,

    input wire                          valid_in, 
    input wire [`RASTER_DIM_BITS-1:0]   x_loc_in,
    input wire [`RASTER_DIM_BITS-1:0]   y_loc_in,
    input wire [`RASTER_PID_BITS-1:0]   pid_in,
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_in,
    output wire                         ready_in,
    
    
     // Outputs
    output wire                         valid_out,
    output wire [OUTPUT_QUADS-1:0]      mask_out,    
    output raster_stamp_t [OUTPUT_QUADS-1:0] stamps_out,
    input  wire                         ready_out
);

    localparam NUM_QUADS_DIM   = 1 << (BLOCK_LOGSIZE - 1);
    localparam PER_BLOCK_QUADS = NUM_QUADS_DIM * NUM_QUADS_DIM;
    localparam OUTPUT_BATCHES  = (PER_BLOCK_QUADS + OUTPUT_QUADS - 1) / OUTPUT_QUADS;
    localparam OUTPUT_SEL_BITS = `LOG2UP(OUTPUT_BATCHES);
    localparam FIFO_DATA_WIDTH = OUTPUT_QUADS + OUTPUT_QUADS * $bits(raster_stamp_t);

    wire stall;

    wire valid_r;
    wire [`RASTER_PID_BITS-1:0] pid_r;
    wire [PER_BLOCK_QUADS-1:0][`RASTER_DIM_BITS-1:0] quad_x_loc, quad_x_loc_r;
    wire [PER_BLOCK_QUADS-1:0][`RASTER_DIM_BITS-1:0] quad_y_loc, quad_y_loc_r;        
    wire [PER_BLOCK_QUADS-1:0][2:0][2:0][`RASTER_DATA_BITS-1:0] quad_edges, quad_edges_r;
    
    // Per-quad edge evaluation
    for (genvar i = 0; i < PER_BLOCK_QUADS; ++i) begin
        integer ii = i % NUM_QUADS_DIM;
        integer jj = i / NUM_QUADS_DIM;
        assign quad_x_loc[i] = x_loc_in + `RASTER_DIM_BITS'(2 * ii);
        assign quad_y_loc[i] = y_loc_in + `RASTER_DIM_BITS'(2 * jj);
        wire [2:0][`RASTER_DATA_BITS-1:0] quad_edge_eval;
        for (genvar k = 0; k < 3; ++k) begin
            assign quad_edge_eval[k] = ii * 2 * edges_in[k][0] + jj * 2 * edges_in[k][1] + edges_in[k][2];
        end
        `EDGE_UPDATE (quad_edges[i], edges_in, quad_edge_eval);
    end

    VX_pipe_register #(
        .DATAW  (1 + `RASTER_PID_BITS + PER_BLOCK_QUADS * (2 * `RASTER_DIM_BITS + 9 * `RASTER_DATA_BITS)),
        .RESETW (1)   
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in, pid_in, quad_x_loc,   quad_y_loc,   quad_edges}),
        .data_out ({valid_r,  pid_r,  quad_x_loc_r, quad_y_loc_r, quad_edges_r})
    );

    wire qe_empty;
    wire [PER_BLOCK_QUADS-1:0]  qe_valid;    
    wire [`RASTER_PID_BITS-1:0] qe_pid;
    wire [PER_BLOCK_QUADS-1:0][3:0] qe_mask;
    wire [PER_BLOCK_QUADS-1:0][`RASTER_DIM_BITS-1:0] qe_x_loc;
    wire [PER_BLOCK_QUADS-1:0][`RASTER_DIM_BITS-1:0] qe_y_loc;    
    wire [PER_BLOCK_QUADS-1:0][3:0][2:0][`RASTER_DATA_BITS-1:0] qe_bcoords;    
    
    VX_raster_qe #(
        .SLICE_ID  (SLICE_ID),
        .NUM_QUADS (PER_BLOCK_QUADS)
    ) quad_evaluator (
        .clk        (clk),
        .reset      (reset),

        .dcrs       (dcrs),

        .empty      (qe_empty),
        
        .enable     (~stall),

        .valid_in   (valid_r),
        .pid_in     (pid_r),
        .x_loc_in   (quad_x_loc_r),
        .y_loc_in   (quad_y_loc_r),
        .edges_in   (quad_edges_r),

        .valid_out  (qe_valid),
        .pid_out    (qe_pid),
        .mask_out   (qe_mask),
        .x_loc_out  (qe_x_loc),
        .y_loc_out  (qe_y_loc),                
        .bcoords_out(qe_bcoords)
    );

    // Populate fifo inputs
    
    reg [OUTPUT_BATCHES-1:0][OUTPUT_QUADS-1:0] fifo_mask_in;
    raster_stamp_t [OUTPUT_BATCHES-1:0][OUTPUT_QUADS-1:0] fifo_stamp_in;
        
    for (genvar i = 0; i < OUTPUT_BATCHES * OUTPUT_QUADS; ++i) begin
        integer q = i % OUTPUT_QUADS;
        integer b = i / OUTPUT_QUADS;
        always @(*) begin
            fifo_mask_in[b][q]  = 0;
            fifo_stamp_in[b][q] = 'x;
            if (i < PER_BLOCK_QUADS) begin
                fifo_mask_in [b][q]         = qe_valid[i];
                fifo_stamp_in[b][q].pos_x   = qe_x_loc[i][`RASTER_DIM_BITS-1:1];
                fifo_stamp_in[b][q].pos_y   = qe_y_loc[i][`RASTER_DIM_BITS-1:1];
                fifo_stamp_in[b][q].mask    = qe_mask[i];
                fifo_stamp_in[b][q].pid     = qe_pid;
                fifo_stamp_in[b][q].bcoords = qe_bcoords[i];
            end            
        end
    end

    // output batch select        

    wire [OUTPUT_BATCHES-1:0] batch_valid;
    reg [OUTPUT_BATCHES-1:0]  batch_sent;
    wire [OUTPUT_BATCHES-1:0] batch_sent_n;

    wire [OUTPUT_SEL_BITS-1:0] fifo_arb_index;
    wire [OUTPUT_BATCHES-1:0]  fifo_arb_onehot;

    wire fifo_push, fifo_pop;
    wire fifo_full, fifo_empty;
    
    for (genvar i = 0; i < OUTPUT_BATCHES; ++i) begin
        assign batch_valid[i] = (| fifo_mask_in[i]);
    end

    assign batch_sent_n = batch_sent | (fifo_push ? fifo_arb_onehot : OUTPUT_BATCHES'(0));

    wire batch_sent_all = (batch_sent_n == batch_valid);

    VX_priority_arbiter #(
        .NUM_REQS (OUTPUT_BATCHES)
    ) fifo_arbiter (
        .clk          (clk),
        .reset        (reset),        
        `UNUSED_PIN   (unlock),
        .requests     (batch_valid & ~batch_sent),
        .grant_index  (fifo_arb_index),
        .grant_onehot (fifo_arb_onehot),
        `UNUSED_PIN   (grant_valid)
    );

    always @(posedge clk) begin
        if (reset) begin
            batch_sent <= 0;
        end else begin
            if (fifo_push) begin
                if (batch_sent_all) begin
                    batch_sent <= 0;
                end else begin
                    batch_sent <= batch_sent_n;
                end
            end
        end
    end

    // fifo queue

    wire fifo_valid_in = (| fifo_mask_in[fifo_arb_index]);

    assign fifo_push = fifo_valid_in && ~fifo_full;

    assign fifo_pop = valid_out && ready_out;

    VX_fifo_queue #(
        .DATAW	 (FIFO_DATA_WIDTH),
        .SIZE    (QUAD_FIFO_DEPTH),
        .OUT_REG (1)
    ) fifo_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (fifo_push),
        .pop        (fifo_pop),
        .data_in    ({fifo_mask_in[fifo_arb_index], fifo_stamp_in[fifo_arb_index]}),
        .data_out   ({mask_out,                     stamps_out}),
        .full       (fifo_full),
        .empty      (fifo_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    assign valid_out = ~fifo_empty;

    assign stall = fifo_valid_in && (fifo_full || ~batch_sent_all);

    assign ready_in = ~stall;

    assign empty = ~valid_r 
                && qe_empty
                && fifo_empty;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (valid_in && ready_in) begin
            `TRACE(2, ("%d: raster-be-in: x=%0d, y=%0d, pid=%0d, edge={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                $time, x_loc_in, y_loc_in, pid_in,
                edges_in[0][0], edges_in[0][1], edges_in[0][2],
                edges_in[1][0], edges_in[1][1], edges_in[1][2],
                edges_in[2][0], edges_in[2][1], edges_in[2][2]));
        end
        
        for (integer i = 0; i < OUTPUT_QUADS; ++i) begin
            if (valid_out && ready_out) begin
                `TRACE(2, ("%d: raster-be-out[%0d]: x=%0d, y=%0d, mask=%0d, pid=%0d, bcoords={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                    $time, i, stamps_out[i].pos_x, stamps_out[i].pos_y, mask_out[i], stamps_out[i].pid,
                    stamps_out[i].bcoords[0][0], stamps_out[i].bcoords[0][1], stamps_out[i].bcoords[0][2], 
                    stamps_out[i].bcoords[1][0], stamps_out[i].bcoords[1][1], stamps_out[i].bcoords[1][2], 
                    stamps_out[i].bcoords[2][0], stamps_out[i].bcoords[2][1], stamps_out[i].bcoords[2][2], 
                    stamps_out[i].bcoords[3][0], stamps_out[i].bcoords[3][1], stamps_out[i].bcoords[3][2]));
            end
        end
    end
`endif

endmodule

// Block Evaluator
// Functionality: Receive a block of NxN quads.
// 1. Breaks it into quads and perform quad evaluation.
// 2. Return overlapped quads.

`include "VX_raster_define.vh"

module VX_raster_be #(
    parameter `STRING INSTANCE_ID = "",
    parameter BLOCK_LOGSIZE   = 5,
    parameter OUTPUT_QUADS    = 2,
    parameter QUAD_FIFO_DEPTH = 4    
) (
    // Standard inputs
    input wire clk,
    input wire reset,

    // Device configurations
    raster_dcrs_t dcrs,

    input wire                          valid_in, 
    input wire [`RASTER_DIM_BITS-1:0]   xloc_in,
    input wire [`RASTER_DIM_BITS-1:0]   yloc_in,
    input wire [`RASTER_DIM_BITS-1:0]   xmin_in,
    input wire [`RASTER_DIM_BITS-1:0]   xmax_in,
    input wire [`RASTER_DIM_BITS-1:0]   ymin_in,   
    input wire [`RASTER_DIM_BITS-1:0]   ymax_in,
    input wire [`RASTER_PID_BITS-1:0]   pid_in,
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_in,
    output wire                         ready_in,    
    
     // Outputs
    output wire                         valid_out,
    output raster_stamp_t [OUTPUT_QUADS-1:0] stamps_out,
    output wire                         busy_out,
    input  wire                         ready_out
);

    localparam NUM_QUADS_DIM   = 1 << (BLOCK_LOGSIZE - 1);
    localparam PER_BLOCK_QUADS = NUM_QUADS_DIM * NUM_QUADS_DIM;
    localparam OUTPUT_BATCHES  = (PER_BLOCK_QUADS + OUTPUT_QUADS - 1) / OUTPUT_QUADS;
    localparam OUTPUT_SEL_BITS = `LOG2UP(OUTPUT_BATCHES);
    localparam FIFO_DATA_WIDTH = OUTPUT_QUADS * $bits(raster_stamp_t);

    wire stall;

    wire valid_r;
    wire [`RASTER_PID_BITS-1:0] pid_r;
    wire [PER_BLOCK_QUADS-1:0][`RASTER_DIM_BITS-1:0] quad_xloc, quad_xloc_r;
    wire [PER_BLOCK_QUADS-1:0][`RASTER_DIM_BITS-1:0] quad_yloc, quad_yloc_r;        
    wire [PER_BLOCK_QUADS-1:0][2:0][2:0][`RASTER_DATA_BITS-1:0] quad_edges, quad_edges_r;
    
    // Per-quad edge evaluation
    for (genvar i = 0; i < PER_BLOCK_QUADS; ++i) begin
        localparam ii = i % NUM_QUADS_DIM;
        localparam jj = i / NUM_QUADS_DIM;
        assign quad_xloc[i] = xloc_in + `RASTER_DIM_BITS'(2 * ii);
        assign quad_yloc[i] = yloc_in + `RASTER_DIM_BITS'(2 * jj);
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
        .data_in  ({valid_in, pid_in, quad_xloc,   quad_yloc,   quad_edges}),
        .data_out ({valid_r,  pid_r,  quad_xloc_r, quad_yloc_r, quad_edges_r})
    );

    wire qe_valid;
    wire [PER_BLOCK_QUADS-1:0]  qe_overlap;    
    wire [`RASTER_PID_BITS-1:0] qe_pid;
    wire [PER_BLOCK_QUADS-1:0][3:0] qe_mask;
    wire [PER_BLOCK_QUADS-1:0][`RASTER_DIM_BITS-1:0] qe_xloc;
    wire [PER_BLOCK_QUADS-1:0][`RASTER_DIM_BITS-1:0] qe_yloc;    
    wire [PER_BLOCK_QUADS-1:0][2:0][3:0][`RASTER_DATA_BITS-1:0] qe_bcoords;
    
    VX_raster_qe #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_QUADS   (PER_BLOCK_QUADS)
    ) quad_evaluator (
        .clk        (clk),
        .reset      (reset),

        .dcrs       (dcrs),
                        
        .enable     (~stall),

        .valid_in   (valid_r),
        .pid_in     (pid_r),
        .xloc_in    (quad_xloc_r),
        .yloc_in    (quad_yloc_r),
        .xmin_in    (xmin_in),
        .xmax_in    (xmax_in),
        .ymin_in    (ymin_in),
        .ymax_in    (ymax_in),        
        .edges_in   (quad_edges_r),

        .valid_out  (qe_valid),
        .overlap_out(qe_overlap),
        .pid_out    (qe_pid),
        .mask_out   (qe_mask),
        .xloc_out   (qe_xloc),
        .yloc_out   (qe_yloc),                
        .bcoords_out(qe_bcoords)
    );

    // Populate fifo inputs
    
    wire [OUTPUT_BATCHES-1:0][OUTPUT_QUADS-1:0] fifo_mask_in;
    raster_stamp_t [OUTPUT_BATCHES-1:0][OUTPUT_QUADS-1:0] fifo_stamp_in;
        
    for (genvar i = 0; i < OUTPUT_BATCHES * OUTPUT_QUADS; ++i) begin
        localparam q = i % OUTPUT_QUADS;
        localparam b = i / OUTPUT_QUADS;
        if (i < PER_BLOCK_QUADS) begin
            assign fifo_mask_in [b][q]         = qe_overlap[i];
            assign fifo_stamp_in[b][q].pos_x   = qe_xloc[i][`RASTER_DIM_BITS-1:1];
            assign fifo_stamp_in[b][q].pos_y   = qe_yloc[i][`RASTER_DIM_BITS-1:1];
            assign fifo_stamp_in[b][q].mask    = qe_mask[i];
            assign fifo_stamp_in[b][q].pid     = qe_pid;
            assign fifo_stamp_in[b][q].bcoords = qe_bcoords[i];
        end else begin
            assign fifo_mask_in[b][q]  = 0;
            assign fifo_stamp_in[b][q] = '0;
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
        assign batch_valid[i] = qe_valid && (| fifo_mask_in[i]);
    end

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

    assign batch_sent_n = batch_sent | ({OUTPUT_BATCHES{fifo_push}} & fifo_arb_onehot);

    wire batch_sent_all = (batch_sent_n == batch_valid);

    always @(posedge clk) begin
        if (reset) begin
            batch_sent <= '0;
        end else begin
            if (fifo_push) begin
                if (batch_sent_all) begin
                    batch_sent <= '0;
                end else begin
                    batch_sent <= batch_sent_n;
                end
            end
        end
    end

    // fifo queue

    wire fifo_valid_in = qe_valid && (| fifo_mask_in[fifo_arb_index]);

    wire [FIFO_DATA_WIDTH-1:0] fifo_data_in = fifo_stamp_in[fifo_arb_index];

    assign fifo_push = fifo_valid_in && ~fifo_full;

    assign fifo_pop = valid_out && ready_out;

    VX_fifo_queue #(
        .DATAW (FIFO_DATA_WIDTH),
        .DEPTH (QUAD_FIFO_DEPTH)
    ) fifo_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (fifo_push),
        .pop        (fifo_pop),
        .data_in    (fifo_data_in),
        .data_out   (stamps_out),
        .full       (fifo_full),
        .empty      (fifo_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    assign valid_out = ~fifo_empty;

    assign stall = fifo_valid_in && (fifo_full || ~batch_sent_all);

    assign ready_in = ~stall;

    assign busy_out = valid_r || qe_valid;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (valid_in && ready_in) begin
            `TRACE(2, ("%d: %s-be-in: x=%0d, y=%0d, pid=%0d, edge={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                $time, INSTANCE_ID, xloc_in, yloc_in, pid_in,
                edges_in[0][0], edges_in[0][1], edges_in[0][2],
                edges_in[1][0], edges_in[1][1], edges_in[1][2],
                edges_in[2][0], edges_in[2][1], edges_in[2][2]));
        end
        
        for (integer i = 0; i < OUTPUT_QUADS; ++i) begin
            if (valid_out && ready_out) begin
                `TRACE(2, ("%d: %s-be-out[%0d]: x=%0d, y=%0d, mask=%0d, pid=%0d, bcoords={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                    $time, INSTANCE_ID, i, stamps_out[i].pos_x, stamps_out[i].pos_y, stamps_out[i].mask, stamps_out[i].pid,
                    stamps_out[i].bcoords[0][0], stamps_out[i].bcoords[1][0], stamps_out[i].bcoords[2][0], 
                    stamps_out[i].bcoords[0][1], stamps_out[i].bcoords[1][1], stamps_out[i].bcoords[2][1], 
                    stamps_out[i].bcoords[0][2], stamps_out[i].bcoords[1][2], stamps_out[i].bcoords[2][2], 
                    stamps_out[i].bcoords[0][3], stamps_out[i].bcoords[1][3], stamps_out[i].bcoords[2][3]));
            end
        end
    end
`endif

endmodule

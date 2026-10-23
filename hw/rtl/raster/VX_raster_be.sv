// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Block Evaluator
// Functionality: Receive a block of NxN quads.
// 1. Breaks it into quads and perform quad evaluation.
// 2. Return overlapped quads.

`include "VX_raster_define.vh"

module VX_raster_be import VX_raster_pkg::*; #(
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
    input wire [`VX_RASTER_DIM_BITS-1:0] xloc_in,
    input wire [`VX_RASTER_DIM_BITS-1:0] yloc_in,
    input raster_rect_t                 scissor_in,
    input wire [`VX_RASTER_PID_BITS-1:0] pid_in,
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_in,
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    input wire [2:0][`RASTER_DATA_BITS-1:0] zplane_in,
`endif
    output wire                         ready_in,

     // Outputs
    output wire                         valid_out,
    output raster_stamp_t [OUTPUT_QUADS-1:0] stamps_out,
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Depth plane carried through with each emitted wave (early-Z).
    output wire [2:0][`RASTER_DATA_BITS-1:0] zplane_out,
`endif
    output wire                         busy_out,
    input  wire                         ready_out
);

    localparam NUM_QUADS_DIM   = 1 << (BLOCK_LOGSIZE - 1);
    localparam PER_BLOCK_QUADS = NUM_QUADS_DIM * NUM_QUADS_DIM;
    localparam OUTPUT_BATCHES  = (PER_BLOCK_QUADS + OUTPUT_QUADS - 1) / OUTPUT_QUADS;
    localparam OUTPUT_SEL_BITS = `LOG2UP(OUTPUT_BATCHES);
    localparam FIFO_DATA_WIDTH = OUTPUT_QUADS * $bits(raster_stamp_t);
    localparam SAMPLE_BITS     = `CLOG2(RASTER_SAMPLES);

    wire stall;

    // Under multisampling a block is evaluated once per sample. It is held at the
    // input: its first cycle computes the block's origin term moved to sample 0,
    // and on each of the next RASTER_SAMPLES cycles that registered term of sample
    // k stands in for the origin term of every quad -- so the quad evaluator's
    // per-pixel tests give sample k's coverage bit -- while sample k+1's is
    // computed. The four bits assemble after the evaluator.
    wire msaa = dcrs.msaa;
    reg  [SAMPLE_BITS:0]   phase_r;   // 0: sample 0's term; k+1: sample k evaluated
    reg  [2:0][`RASTER_DATA_BITS-1:0] corg_r;
    wire [SAMPLE_BITS-1:0] sample_in      = SAMPLE_BITS'(phase_r - (SAMPLE_BITS+1)'(1));
    wire [SAMPLE_BITS-1:0] offset_sample  = SAMPLE_BITS'(phase_r);
    wire                   eval_in        = ~msaa || (phase_r != '0);
    wire                   last_sample_in = ~msaa || (phase_r == (SAMPLE_BITS+1)'(RASTER_SAMPLES));

    `STATIC_ASSERT((RASTER_SAMPLES == 4), ("the sample offsets are the 4x pattern's"))

    wire [2:0][`RASTER_DATA_BITS-1:0] corg_n;
    for (genvar k = 0; k < 3; ++k) begin : g_corg
        assign corg_n[k] = edges_in[k][2] + raster_sample_offset(edges_in[k][0], edges_in[k][1], offset_sample);
    end

    always @(posedge clk) begin
        if (reset) begin
            phase_r <= '0;
        end else if (valid_in && ~stall && msaa) begin
            phase_r <= last_sample_in ? '0 : (phase_r + (SAMPLE_BITS+1)'(1));
        end
        if (valid_in && ~stall) begin
            corg_r <= corg_n;
        end
    end

    wire valid_r;
    wire [SAMPLE_BITS-1:0] sample_r;
    wire [`VX_RASTER_PID_BITS-1:0] pid_r;
    raster_rect_t scissor_r;
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    wire [2:0][`RASTER_DATA_BITS-1:0] zplane_r;
`endif
    wire [PER_BLOCK_QUADS-1:0][`VX_RASTER_DIM_BITS-1:0] quad_xloc, quad_xloc_r;
    wire [PER_BLOCK_QUADS-1:0][`VX_RASTER_DIM_BITS-1:0] quad_yloc, quad_yloc_r;
    wire [PER_BLOCK_QUADS-1:0][2:0][2:0][`RASTER_DATA_BITS-1:0] quad_edges, quad_edges_r;

    // Per-quad edge evaluation
    for (genvar i = 0; i < PER_BLOCK_QUADS; ++i) begin : g_quad_edges
        localparam ii = i % NUM_QUADS_DIM;
        localparam jj = i / NUM_QUADS_DIM;
        assign quad_xloc[i] = xloc_in + `VX_RASTER_DIM_BITS'(2 * ii);
        assign quad_yloc[i] = yloc_in + `VX_RASTER_DIM_BITS'(2 * jj);
        wire [2:0][`RASTER_DATA_BITS-1:0] quad_edge_eval;
        for (genvar k = 0; k < 3; ++k) begin : g_k
            wire [`RASTER_DATA_BITS-1:0] corg = msaa ? corg_r[k] : edges_in[k][2];
            assign quad_edge_eval[k] = ii * 2 * edges_in[k][0] + jj * 2 * edges_in[k][1] + corg;
        end
        `EDGE_UPDATE (quad_edges[i], edges_in, quad_edge_eval);
    end

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Carry the constant depth plane through the block-eval pipe with the quads.
    VX_pipe_register #(
        .DATAW  (1 + SAMPLE_BITS + `VX_RASTER_PID_BITS + RASTER_RECT_BITS + 3 * `RASTER_DATA_BITS + PER_BLOCK_QUADS * (2 * `VX_RASTER_DIM_BITS + 9 * `RASTER_DATA_BITS)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in && eval_in, sample_in, pid_in, scissor_in, zplane_in, quad_xloc, quad_yloc, quad_edges}),
        .data_out ({valid_r,  sample_r,  pid_r,  scissor_r,  zplane_r,  quad_xloc_r, quad_yloc_r, quad_edges_r})
    );
`else
    VX_pipe_register #(
        .DATAW  (1 + SAMPLE_BITS + `VX_RASTER_PID_BITS + RASTER_RECT_BITS + PER_BLOCK_QUADS * (2 * `VX_RASTER_DIM_BITS + 9 * `RASTER_DATA_BITS)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in && eval_in, sample_in, pid_in, scissor_in, quad_xloc, quad_yloc, quad_edges}),
        .data_out ({valid_r,  sample_r,  pid_r,  scissor_r,  quad_xloc_r, quad_yloc_r, quad_edges_r})
    );
`endif

    wire qe_valid;
    wire [`VX_RASTER_PID_BITS-1:0] qe_pid;
    wire [PER_BLOCK_QUADS-1:0][3:0] qe_mask;
    wire [PER_BLOCK_QUADS-1:0][`VX_RASTER_DIM_BITS-1:0] qe_xloc;
    wire [PER_BLOCK_QUADS-1:0][`VX_RASTER_DIM_BITS-1:0] qe_yloc;

    VX_raster_qe #(
        .INSTANCE_ID ($sformatf("%s-qe", INSTANCE_ID)),
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
        .xmin_in    (scissor_r.xmin),
        .xmax_in    (scissor_r.xmax),
        .ymin_in    (scissor_r.ymin),
        .ymax_in    (scissor_r.ymax),
        .edges_in   (quad_edges_r),

        .valid_out  (qe_valid),
        .pid_out    (qe_pid),
        .mask_out   (qe_mask),
        .xloc_out   (qe_xloc),
        .yloc_out   (qe_yloc)
    );

    // The quad evaluator registers its outputs once; the sample index follows it.
    wire [SAMPLE_BITS-1:0] qe_sample;

    VX_pipe_register #(
        .DATAW  (SAMPLE_BITS)
    ) qe_sample_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  (sample_r),
        .data_out (qe_sample)
    );

    // A pixel's sample mask: the bits of samples 0..2 accumulate here while their
    // evaluations go past, and the block leaves with its last sample's. At one
    // sample a pixel's coverage is bit 0.
    reg  [PER_BLOCK_QUADS-1:0][3:0][RASTER_SAMPLES-2:0] smask_acc_r;
    wire [PER_BLOCK_QUADS-1:0][3:0][RASTER_SAMPLES-1:0] qe_smask;
    wire [PER_BLOCK_QUADS-1:0] qe_covered;
    wire qe_last = ~msaa || (qe_sample == SAMPLE_BITS'(RASTER_SAMPLES - 1));
    wire qe_valid_block = qe_valid && qe_last;

    for (genvar i = 0; i < PER_BLOCK_QUADS; ++i) begin : g_smask
        for (genvar p = 0; p < 4; ++p) begin : g_p
            assign qe_smask[i][p] = msaa ? {qe_mask[i][p], smask_acc_r[i][p]}
                                         : RASTER_SAMPLES'(qe_mask[i][p]);
        end
        assign qe_covered[i] = (| qe_smask[i]);
    end

    always @(posedge clk) begin
        if (qe_valid && ~stall && ~qe_last) begin
            for (integer i = 0; i < PER_BLOCK_QUADS; ++i) begin
                for (integer p = 0; p < 4; ++p) begin
                    smask_acc_r[i][p][qe_sample] <= qe_mask[i][p];
                end
            end
        end
    end

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // The quad evaluator registers its outputs once; delay the plane the same
    // way so each wave leaves with its own primitive's plane, not the plane of
    // the block behind it.
    wire [2:0][`RASTER_DATA_BITS-1:0] qe_zplane;

    VX_pipe_register #(
        .DATAW  (3 * `RASTER_DATA_BITS)
    ) qe_zplane_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  (zplane_r),
        .data_out (qe_zplane)
    );
`endif

    // Populate fifo inputs

    wire [OUTPUT_BATCHES-1:0][OUTPUT_QUADS-1:0] fifo_mask_in;
    raster_stamp_t [OUTPUT_BATCHES-1:0][OUTPUT_QUADS-1:0] fifo_stamp_in;

    for (genvar i = 0; i < OUTPUT_BATCHES * OUTPUT_QUADS; ++i) begin : g_fifo_inputs
        localparam q = i % OUTPUT_QUADS;
        localparam b = i / OUTPUT_QUADS;
        if (i < PER_BLOCK_QUADS) begin : g_valid
            assign fifo_mask_in [b][q]         = qe_covered[i];
            assign fifo_stamp_in[b][q].pos_x   = qe_xloc[i][`VX_RASTER_DIM_BITS-1:1];
            assign fifo_stamp_in[b][q].pos_y   = qe_yloc[i][`VX_RASTER_DIM_BITS-1:1];
            assign fifo_stamp_in[b][q].mask    = qe_smask[i];
            assign fifo_stamp_in[b][q].pid     = qe_pid;
        end else begin : g_extra
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

    wire fifo_fire;

    for (genvar i = 0; i < OUTPUT_BATCHES; ++i) begin : g_batch_valid
        assign batch_valid[i] = qe_valid_block && (| fifo_mask_in[i]);
    end

    VX_priority_arbiter #(
        .NUM_REQS (OUTPUT_BATCHES)
    ) fifo_arbiter (
        .clk          (clk),
        .reset        (reset),
        .requests     (batch_valid & ~batch_sent),
        .grant_index  (fifo_arb_index),
        .grant_onehot (fifo_arb_onehot),
        `UNUSED_PIN   (grant_valid),
        .grant_ready  (1'b1)
    );

    assign batch_sent_n = batch_sent | ({OUTPUT_BATCHES{fifo_fire}} & fifo_arb_onehot);

    wire batch_sent_all = (batch_sent_n == batch_valid);

    always @(posedge clk) begin
        if (reset) begin
            batch_sent <= '0;
        end else begin
            if (fifo_fire) begin
                if (batch_sent_all) begin
                    batch_sent <= '0;
                end else begin
                    batch_sent <= batch_sent_n;
                end
            end
        end
    end

    // output queue

    wire fifo_valid_in, fifo_ready_in;
    assign fifo_valid_in = qe_valid_block && (| fifo_mask_in[fifo_arb_index]);
    assign fifo_fire = fifo_valid_in && fifo_ready_in;

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    VX_elastic_buffer #(
        .DATAW (FIFO_DATA_WIDTH + 3 * `RASTER_DATA_BITS),
        .SIZE  (QUAD_FIFO_DEPTH)
    ) fifo_queue (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fifo_valid_in),
        .ready_in  (fifo_ready_in),
        .data_in   ({qe_zplane,  fifo_stamp_in[fifo_arb_index]}),
        .data_out  ({zplane_out, stamps_out}),
        .valid_out (valid_out),
        .ready_out (ready_out)
    );
`else
    VX_elastic_buffer #(
        .DATAW (FIFO_DATA_WIDTH),
        .SIZE  (QUAD_FIFO_DEPTH)
    ) fifo_queue (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fifo_valid_in),
        .ready_in  (fifo_ready_in),
        .data_in   (fifo_stamp_in[fifo_arb_index]),
        .data_out  (stamps_out),
        .valid_out (valid_out),
        .ready_out (ready_out)
    );
`endif

    assign stall = fifo_valid_in && (~fifo_ready_in || ~batch_sent_all);

    assign ready_in = ~stall && last_sample_in;

    assign busy_out = valid_r || qe_valid;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (valid_in && ready_in) begin
            `TRACE(2, ("%d: %s-be-in: x=%0d, y=%0d, pid=%0d, edge={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                $time, INSTANCE_ID, xloc_in, yloc_in, pid_in,
                edges_in[0][0], edges_in[0][1], edges_in[0][2],
                edges_in[1][0], edges_in[1][1], edges_in[1][2],
                edges_in[2][0], edges_in[2][1], edges_in[2][2]))
        end

        for (integer i = 0; i < OUTPUT_QUADS; ++i) begin
            if (valid_out && ready_out) begin
                `TRACE(2, ("%d: %s-be-out[%0d]: x=%0d, y=%0d, mask=0x%0h, pid=%0d\n",
                    $time, INSTANCE_ID, i, stamps_out[i].pos_x, stamps_out[i].pos_y, stamps_out[i].mask, stamps_out[i].pid))
            end
        end
    end
`endif

endmodule

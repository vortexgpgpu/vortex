//!/bin/bash

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

// Rasterizer Tile Evaluator
// Functionality: Receive a tile.
// 1. Recursive descend sub-tiles that overlap primitive.
// 2. Stop recursion when a tile reaches block size.
// 3. Return overlapped blocks.

`include "VX_raster_define.vh"

module VX_raster_te #(
    parameter `STRING INSTANCE_ID = "",
    parameter TILE_LOGSIZE  = 5,
    parameter BLOCK_LOGSIZE = 2
) (
    input wire clk,
    input wire reset,

    // Inputs
    input wire                          valid_in,
    input wire [`VX_RASTER_DIM_BITS-1:0] xloc_in,
    input wire [`VX_RASTER_DIM_BITS-1:0] yloc_in,
    input wire [`VX_RASTER_PID_BITS-1:0] pid_in,
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_in,
    input wire [2:0][`RASTER_DATA_BITS-1:0] extents_in,
    output wire                         ready_in,

    // Outputs
    output wire                         valid_out,
    output wire [`VX_RASTER_DIM_BITS-1:0] xloc_out,
    output wire [`VX_RASTER_DIM_BITS-1:0] yloc_out,
    output wire [`VX_RASTER_PID_BITS-1:0] pid_out,
    output wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_out,
    input wire                          ready_out
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam LEVEL_BITS      = (TILE_LOGSIZE - BLOCK_LOGSIZE) + 1;
    localparam TILE_FIFO_DEPTH = 1 << (2 * (TILE_LOGSIZE - BLOCK_LOGSIZE));
    localparam FIFO_DATA_WIDTH = 2 * `VX_RASTER_DIM_BITS + 3 * `RASTER_DATA_BITS + LEVEL_BITS;

    wire stall;

    reg [2:0][`RASTER_DATA_BITS-1:0] tile_extents;
    reg [2:0][2:0][`RASTER_DATA_BITS-1:0] tile_edges;
    reg [`VX_RASTER_PID_BITS-1:0]       tile_pid;
    reg [`VX_RASTER_DIM_BITS-1:0]       tile_xloc;
    reg [`VX_RASTER_DIM_BITS-1:0]       tile_yloc;
    reg [2:0][`RASTER_DATA_BITS-1:0]    tile_edge_eval;
    reg [LEVEL_BITS-1:0]                tile_level;

    wire [`VX_RASTER_DIM_BITS-1:0]      tile_xloc_r;
    wire [`VX_RASTER_DIM_BITS-1:0]      tile_yloc_r;
    wire [2:0][`RASTER_DATA_BITS-1:0]   tile_edge_eval_r;
    wire [LEVEL_BITS-1:0]               tile_level_r;

    wire [3:0][`VX_RASTER_DIM_BITS-1:0] subtile_xloc, subtile_xloc_r;
    wire [3:0][`VX_RASTER_DIM_BITS-1:0] subtile_yloc, subtile_yloc_r;
    wire [3:0][2:0][`RASTER_DATA_BITS-1:0] subtile_edge_eval, subtile_edge_eval_r;
    wire [LEVEL_BITS-1:0] subtile_level, subtile_level_r;

    wire [`VX_RASTER_DIM_BITS-1:0]      fifo_xloc;
    wire [`VX_RASTER_DIM_BITS-1:0]      fifo_yloc;
    wire [2:0][`RASTER_DATA_BITS-1:0]   fifo_edge_eval;
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
                    tile_xloc           <= fifo_xloc;
                    tile_yloc           <= fifo_yloc;
                    tile_edge_eval      <= fifo_edge_eval;
                    tile_level          <= fifo_level;
                end else
                if (is_fifo_bypass) begin
                    // fifo bypass first sub-tile
                    tile_valid          <= 1;
                    tile_xloc           <= subtile_xloc_r[0];
                    tile_yloc           <= subtile_yloc_r[0];
                    tile_edge_eval      <= subtile_edge_eval_r[0];
                    tile_level          <= subtile_level_r;
                end else
                if (valid_in && ~tile_valid) begin
                    // select new tile input
                    tile_valid          <= 1;
                    tile_extents        <= extents_in;
                    tile_edges          <= edges_in;
                    tile_pid            <= pid_in;
                    tile_xloc           <= xloc_in;
                    tile_yloc           <= yloc_in;
                    tile_edge_eval[0]   <= edges_in[0][2];
                    tile_edge_eval[1]   <= edges_in[1][2];
                    tile_edge_eval[2]   <= edges_in[2][2];
                    tile_level          <= '0;
                end
            end
        end
    end

    // Generate sub-tile info
    wire [`VX_RASTER_DIM_BITS-1:0] tile_logsize = `VX_RASTER_DIM_BITS'(TILE_LOGSIZE-1) - `VX_RASTER_DIM_BITS'(tile_level);
    wire is_block = (tile_logsize < `VX_RASTER_DIM_BITS'(BLOCK_LOGSIZE));
    assign subtile_level = tile_level + LEVEL_BITS'(1);
    for (genvar i = 0; i < 2; ++i) begin
        for (genvar j = 0; j < 2; ++j) begin
            assign subtile_xloc[2 * i + j] = tile_xloc + (`VX_RASTER_DIM_BITS'(i) << tile_logsize);
            assign subtile_yloc[2 * i + j] = tile_yloc + (`VX_RASTER_DIM_BITS'(j) << tile_logsize);
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
        .DATAW  (1 + 1 + 4 * (2 * `VX_RASTER_DIM_BITS + 3 * `RASTER_DATA_BITS) + LEVEL_BITS + 2 * `VX_RASTER_DIM_BITS + 3 * `RASTER_DATA_BITS + LEVEL_BITS),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({tile_valid_e, is_block,   subtile_xloc,   subtile_yloc,   subtile_edge_eval,   subtile_level,   tile_xloc,   tile_yloc,   tile_edge_eval,   tile_level}),
        .data_out ({tile_valid_r, is_block_r, subtile_xloc_r, subtile_yloc_r, subtile_edge_eval_r, subtile_level_r, tile_xloc_r, tile_yloc_r, tile_edge_eval_r, tile_level_r})
    );

    wire [3:0] fifo_ready_in, fifo_valid_out;
    wire [3:0][FIFO_DATA_WIDTH-1:0] fifo_data_out;

    wire output_stall = tile_valid_r && is_block_r && ~ready_out;

    for (genvar i = 0; i < 4; ++i) begin
        wire fifo_valid_in = tile_valid_r && ~is_block_r && ~(is_fifo_bypass && i == 0);
        wire fifo_ready_out = fifo_arb_onehot[i] && ~output_stall;

        VX_elastic_buffer #(
            .DATAW	 (FIFO_DATA_WIDTH),
            .SIZE    (TILE_FIFO_DEPTH),
            .OUT_REG (2)
        ) fifo_queue (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (fifo_valid_in),
            .ready_in  (fifo_ready_in[i]),
            .data_in   ({subtile_xloc_r[i], subtile_yloc_r[i], subtile_edge_eval_r[i], subtile_level_r}),
            .data_out  (fifo_data_out[i]),
            .valid_out (fifo_valid_out[i]),
            .ready_out (fifo_ready_out)
        );
    end

    VX_priority_arbiter #(
        .NUM_REQS (4)
    ) fifo_arbiter (
        .requests     (fifo_valid_out),
        .grant_index  (fifo_arb_index),
        .grant_onehot (fifo_arb_onehot),
        .grant_valid  (fifo_arb_valid)
    );

    assign {fifo_xloc, fifo_yloc, fifo_edge_eval, fifo_level} = fifo_data_out[fifo_arb_index];

    // pipeline stall
    wire fifo_stall = tile_valid_r && ~is_block_r && ~(& fifo_ready_in);
    `RUNTIME_ASSERT(~fifo_stall, ("%t: *** tile evaluator fifo overflow", $time));
    assign stall = /*fifo_stall ||*/ output_stall;

    // can accept next input?
    assign ready_in = ~stall           // no pipeline stall
                   && ~tile_valid      // no tile in process
                   && ~fifo_arb_valid  // no fifo input
                   && ~is_fifo_bypass; // no fifo bypass

    assign valid_out = tile_valid_r && is_block_r;
    assign xloc_out  = tile_xloc_r;
    assign yloc_out  = tile_yloc_r;
    assign pid_out   = tile_pid;
    `EDGE_UPDATE (edges_out, tile_edges, tile_edge_eval_r);

    `UNUSED_VAR (tile_level_r)

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (valid_in && ready_in) begin
            `TRACE(2, ("%d: %s-te-in: x=%0d, y=%0d, edge={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}, extents={0x%0h, 0x%0h, 0x%0h}\n",
                $time, INSTANCE_ID, xloc_in, yloc_in,
                edges_in[0][0], edges_in[0][1], edges_in[0][2],
                edges_in[1][0], edges_in[1][1], edges_in[1][2],
                edges_in[2][0], edges_in[2][1], edges_in[2][2],
                extents_in[0],  extents_in[1],  extents_in[2]));
        end
        if (tile_valid && ~stall) begin
            `TRACE(2, ("%d: %s-te-test: pass=%b, block=%b, level=%0d, x=%0d, y=%0d, edge_eval={0x%0h, 0x%0h, 0x%0h}\n",
                $time, INSTANCE_ID, tile_valid_e, is_block, tile_level, tile_xloc, tile_yloc, edge_eval[0], edge_eval[1], edge_eval[2]));
        end
    end
`endif

endmodule

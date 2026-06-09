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

// Rasterizer Processing Element
// Functionality: Receive a tile.
// 1. Perform tile to blocks generation.
// 2. Perform blocks to quas generation.
// 3. Return overlapped quads.

`include "VX_raster_define.vh"

module VX_raster_slice import VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
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
    input wire [`VX_RASTER_DIM_BITS-1:0]            xloc_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]            yloc_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]            xmin_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]            xmax_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]            ymin_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]            ymax_in,
    input wire [`VX_RASTER_PID_BITS-1:0]            pid_in,
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0]    edges_in,
    input wire [2:0][`RASTER_DATA_BITS-1:0]         extents_in,
    output wire                                     ready_in,

    // Outputs
    output wire                                     valid_out,
    output raster_stamp_t [OUTPUT_QUADS-1:0]        stamps_out,
    output wire                                     busy_out,
    input  wire                                     ready_out
);
    localparam NUM_QUADS_DIM   = 1 << (BLOCK_LOGSIZE - 1);
    localparam PER_BLOCK_QUADS = NUM_QUADS_DIM * NUM_QUADS_DIM;
    localparam OUTPUT_BATCHES  = (PER_BLOCK_QUADS + OUTPUT_QUADS - 1) / OUTPUT_QUADS;
    localparam BLOCK_BUF_SIZE  = 2 * OUTPUT_BATCHES;

    wire be_busy;

    wire                        block_valid;
    wire [`VX_RASTER_DIM_BITS-1:0] block_xloc;
    wire [`VX_RASTER_DIM_BITS-1:0] block_yloc;
    wire [`VX_RASTER_PID_BITS-1:0] block_pid;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] block_edges;
    wire                        block_ready;

    VX_raster_te #(
        .INSTANCE_ID   ($sformatf("%s-te", INSTANCE_ID)),
        .TILE_LOGSIZE  (TILE_LOGSIZE),
        .BLOCK_LOGSIZE (BLOCK_LOGSIZE)
    ) tile_evaluator (
        .clk        (clk),
        .reset      (reset),

        .valid_in   (valid_in),
        .xloc_in    (xloc_in),
        .yloc_in    (yloc_in),
        .pid_in     (pid_in),
        .edges_in   (edges_in),
        .extents_in (extents_in),
        .ready_in   (ready_in),

        .valid_out  (block_valid),
        .xloc_out   (block_xloc),
        .yloc_out   (block_yloc),
        .pid_out    (block_pid),
        .edges_out  (block_edges),
        .ready_out  (block_ready)
    );

    wire                        block_valid_b;
    wire [`VX_RASTER_DIM_BITS-1:0] block_xloc_b;
    wire [`VX_RASTER_DIM_BITS-1:0] block_yloc_b;
    wire [`VX_RASTER_PID_BITS-1:0] block_pid_b;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] block_edges_b;
    wire                        block_ready_b;

    VX_elastic_buffer #(
        .DATAW (2 * `VX_RASTER_DIM_BITS + `VX_RASTER_PID_BITS + 9 * `RASTER_DATA_BITS),
        .SIZE  (BLOCK_BUF_SIZE)
    ) block_req_buf (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (block_valid),
        .ready_in   (block_ready),
        .data_in    ({block_xloc,   block_yloc,   block_pid,   block_edges}),
        .data_out   ({block_xloc_b, block_yloc_b, block_pid_b, block_edges_b}),
        .valid_out  (block_valid_b),
        .ready_out  (block_ready_b)
    );

    VX_raster_be #(
        .INSTANCE_ID     ($sformatf("%s-be", INSTANCE_ID)),
        .BLOCK_LOGSIZE   (BLOCK_LOGSIZE),
        .OUTPUT_QUADS    (OUTPUT_QUADS),
        .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH)
    ) block_evaluator (
        .clk        (clk),
        .reset      (reset),

        .dcrs       (dcrs),

        .valid_in   (block_valid_b),
        .xloc_in    (block_xloc_b),
        .yloc_in    (block_yloc_b),
        .xmin_in    (xmin_in),
        .xmax_in    (xmax_in),
        .ymin_in    (ymin_in),
        .ymax_in    (ymax_in),
        .pid_in     (block_pid_b),
        .edges_in   (block_edges_b),
        .ready_in   (block_ready_b),

        .valid_out  (valid_out),
        .stamps_out (stamps_out),
        .busy_out   (be_busy),
        .ready_out  (ready_out)
    );

    assign busy_out = ~ready_in
                   || block_valid
                   || block_valid_b
                   || be_busy;

endmodule

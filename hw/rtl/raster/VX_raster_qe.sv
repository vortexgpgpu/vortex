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

// Rasterizer Quad Evaluator: receives 2x2 quads, tests each pixel against
// all three triangle edges, and emits the overlap mask and barycentric coordinates.

`include "VX_raster_define.vh"

module VX_raster_qe import VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_QUADS = 4
) (
    input wire clk,
    input wire reset,

    // Device configurations
    raster_dcrs_t dcrs,

    input wire                                          enable,

    // Inputs
    input wire                                          valid_in,
    input wire [`VX_RASTER_PID_BITS-1:0]                pid_in,
    input wire [NUM_QUADS-1:0][`VX_RASTER_DIM_BITS-1:0] xloc_in,
    input wire [NUM_QUADS-1:0][`VX_RASTER_DIM_BITS-1:0] yloc_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]                xmin_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]                xmax_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]                ymin_in,
    input wire [`VX_RASTER_DIM_BITS-1:0]                ymax_in,
    input wire [NUM_QUADS-1:0][2:0][2:0][`RASTER_DATA_BITS-1:0] edges_in,

    // Outputs
    output wire                                         valid_out,
    output wire [NUM_QUADS-1:0]                         overlap_out,
    output wire [`VX_RASTER_PID_BITS-1:0]               pid_out,
    output wire [NUM_QUADS-1:0][3:0]                    mask_out,
    output wire [NUM_QUADS-1:0][`VX_RASTER_DIM_BITS-1:0] xloc_out,
    output wire [NUM_QUADS-1:0][`VX_RASTER_DIM_BITS-1:0] yloc_out
);
    `UNUSED_SPARAM (INSTANCE_ID)

    `UNUSED_VAR (dcrs)

    wire [NUM_QUADS-1:0] overlap;
    wire [NUM_QUADS-1:0][2:0][3:0][`RASTER_DATA_BITS-1:0] edge_eval;
    wire [NUM_QUADS-1:0][3:0] overlap_mask;

    // Vulkan top-left fill rule: a sample lying exactly on an edge (edge value
    // == 0) is covered only if that edge is a top or left edge; otherwise the
    // sample belongs to the abutting triangle. An edge is top-left when its
    // gradient A > 0, or A == 0 and B > 0 (A = d/dx, B = d/dy of the edge
    // function). A and B are primitive-constant across the quads (only the
    // per-quad origin term edges_in[.][.][2] varies), so classify once per edge.
    // Guarantees shared edges are covered exactly once (the two triangles see
    // opposite-sign gradients, so exactly one classifies the edge top-left).
    wire [2:0] is_top_left;
    for (genvar k = 0; k < 3; ++k) begin : g_topleft
        wire a_neg  = edges_in[0][k][0][`RASTER_DATA_BITS-1];
        wire a_zero = (edges_in[0][k][0] == '0);
        wire b_neg  = edges_in[0][k][1][`RASTER_DATA_BITS-1];
        wire b_zero = (edges_in[0][k][1] == '0);
        assign is_top_left[k] = (~a_neg && ~a_zero)
                             || (a_zero && ~b_neg && ~b_zero);
    end

     // Check if primitive overlaps current quad
    for (genvar q = 0; q < NUM_QUADS; ++q) begin : g_overlap_mask
        for (genvar i = 0; i < 2; ++i) begin : g_i
            for (genvar j = 0; j < 2; ++j) begin : g_j
                wire [2:0] edge_fail;
                for (genvar k = 0; k < 3; ++k) begin : g_k
                    assign edge_eval[q][k][2 * j + i] = i * edges_in[q][k][0] + j * edges_in[q][k][1] + edges_in[q][k][2];
                    // Fail when the edge value is negative, or zero on a
                    // non-top-left edge (excluded boundary sample).
                    assign edge_fail[k] = edge_eval[q][k][2 * j + i][`RASTER_DATA_BITS-1]
                                       || (~is_top_left[k] && (edge_eval[q][k][2 * j + i] == '0));
                end
                wire [`VX_RASTER_DIM_BITS-1:0] quad_x = xloc_in[q] | i;
                wire [`VX_RASTER_DIM_BITS-1:0] quad_y = yloc_in[q] | j;
                assign overlap_mask[q][2 * j + i] = ~(| edge_fail)
                                                  && (quad_x >= xmin_in)
                                                  && (quad_x <  xmax_in)
                                                  && (quad_y >= ymin_in)
                                                  && (quad_y <  ymax_in);
            end
        end

        assign overlap[q] = (| overlap_mask[q]);
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_QUADS + `VX_RASTER_PID_BITS + NUM_QUADS * (4 + 2 * `VX_RASTER_DIM_BITS)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  overlap,     pid_in,  overlap_mask, xloc_in,  yloc_in}),
        .data_out ({valid_out, overlap_out, pid_out, mask_out,     xloc_out, yloc_out})
    );

endmodule

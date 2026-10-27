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

`include "VX_define.vh"

// Per-warp CTA state for one group of warps, addressed by group-local warp id:
// the rank, lane record and CTA id of the group's own warps. Written by the
// dispatcher's broadcast; the parent presents only writes that belong to this
// group. The CTA contexts every warp may read are held core-wide.
module VX_csr_cta import VX_gpu_pkg::*; #(
    parameter NUM_WARPS = 1,
    parameter LW_WIDTH  = `LOG2UP(NUM_WARPS)
) (
    input wire                  clk,
    input wire                  reset,

    // this group's warps
    input wire                  warp_write,
    input wire [LW_WIDTH-1:0]   warp_waddr,
    input cta_warp_t            warp_wdata,
    input wire                  id_write,
    input wire [LW_WIDTH-1:0]   id_waddr,
    input wire [NCTA_WIDTH-1:0] id_wdata,

    // CSR read: rdata is valid one cycle after the address, which the CSR unit
    // holds stable for that cycle
    input wire [LW_WIDTH-1:0]   rd_lid,
    output wire [NW_WIDTH-1:0]  rd_cta_rank,
    output cta_lane_t [`VX_CFG_NUM_THREADS-1:0] rd_lane,

    // CTA id of a warp, combinational
    input wire [LW_WIDTH-1:0]   sched_lid,
    output wire [NCTA_WIDTH-1:0] sched_cta_id
);
    cta_warp_t warp_rdata;

    VX_dp_ram #(
        .DATAW     ($bits(cta_warp_t)),
        .SIZE      (NUM_WARPS),
        .RDW_MODE  ("R"),
        .OUT_REG   (1)
    ) cta_warp_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (warp_write),
        .wren  (1'b1),
        .waddr (warp_waddr),
        .wdata (warp_wdata),
        .raddr (rd_lid),
        .rdata (warp_rdata)
    );

    assign rd_cta_rank = warp_rdata.cta_rank;
    assign rd_lane     = warp_rdata.lane_launch;

    reg [NUM_WARPS-1:0][NCTA_WIDTH-1:0] cta_id_r;
    always @(posedge clk) begin
        if (id_write) begin
            cta_id_r[id_waddr] <= id_wdata;
        end
    end
    assign sched_cta_id = cta_id_r[sched_lid];

endmodule

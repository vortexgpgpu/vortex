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

module VX_commit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_commit_if.slave      commit_if [NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH],

    // outputs
    VX_writeback_if.master  writeback_if  [`VX_CFG_ISSUE_WIDTH],
    VX_commit_sched_if.master commit_sched_if
);
    for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_slices
        VX_commit_if slice_commit_if[NUM_EX_UNITS]();
        for (genvar j = 0; j < NUM_EX_UNITS; ++j) begin : g_commit_if
            `ASSIGN_VX_IF(slice_commit_if[j], commit_if[j * `VX_CFG_ISSUE_WIDTH + i]);
        end

        wire [PER_ISSUE_WARPS-1:0] committed_wis;

        VX_commit_slice #(
            .INSTANCE_ID (INSTANCE_ID)
        ) commit_slice (
            .clk           (clk),
            .reset         (reset),
            .commit_if     (slice_commit_if),
            .writeback_if  (writeback_if[i]),
            .committed_wis (committed_wis)
        );

        // wid = {wis, isw}: slice i owns every ISSUE_WIDTH-th warp starting at i
        for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_committed
            assign commit_sched_if.committed_warps[w * `VX_CFG_ISSUE_WIDTH + i] = committed_wis[w];
        end
    end

endmodule

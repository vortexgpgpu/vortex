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

// Per-warp instruction ids. NUM_WARPS is the number of warps counted here: all
// of the core's, or the PER_FETCH_WARPS of one frontend slice (wid = {wfs, slice}).
module VX_uuid_gen import VX_gpu_pkg::*; #(
    parameter NUM_WARPS = `VX_CFG_NUM_WARPS
) (
    input wire clk,
    input wire reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,
    input wire incr,
    input wire [NW_WIDTH-1:0] wid,
    output wire [UUID_WIDTH-1:0] uuid
);
    localparam GNW_WIDTH = UUID_WIDTH - 32;
    localparam LW = `LOG2UP(NUM_WARPS);

    wire [LW-1:0] lid;
    if (NUM_WARPS == `VX_CFG_NUM_WARPS) begin : g_core_warps
        assign lid = LW'(wid);
    end else begin : g_slice_warps
        `STATIC_ASSERT (NUM_WARPS == PER_FETCH_WARPS, ("invalid parameter"))
        assign lid = wid_to_wfs(wid);
    end

    reg [31:0] uuid_cntrs [0:NUM_WARPS-1];
    reg [NUM_WARPS-1:0] has_uuid_cntrs;

    always @(posedge clk) begin
        if (reset) begin
            has_uuid_cntrs <= '0;
        end else if (incr) begin
            has_uuid_cntrs[lid] <= 1;
        end
        if (incr) begin
            uuid_cntrs[lid] <= has_uuid_cntrs[lid] ? (uuid_cntrs[lid] + 1) : 1;
        end
    end

    wire [GNW_WIDTH-1:0] g_wid = (GNW_WIDTH'(core_id) << NW_BITS) + GNW_WIDTH'(wid);
    assign uuid = {g_wid, (has_uuid_cntrs[lid] ? uuid_cntrs[lid] : 0)};

endmodule

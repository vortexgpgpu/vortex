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

// One box-result collector entry: its busy/kind state, the result counts, and
// the hit children kept sorted by ascending t as the box PE delivers them.
module VX_rtu_coll_entry import VX_rtu_pkg::*; #(
    parameter NODE_W = RTU_NODE_W
) (
    input  wire                           clk,
    input  wire                           reset,

    // allocation and release of this entry
    input  wire                           alloc,
    input  wire                           alloc_proc,
    input  wire                           free,

    // box PE result, and whether it belongs to this entry
    input  wire                           box_valid,
    input  wire                           box_sel,
    input  wire                           box_hit,
    input  wire                           box_pushable,
    input  wire [31:0]                    box_t_near,
    input  wire [31:0]                    box_childoff,
    // the result's t against each slot's t, taken a cycle ahead
    input  wire [NODE_W-1:0]              ins_t_le,

    output reg                            busy,
    output reg                            proc,
    output reg  [RTU_CHILD_BITS-1:0]      cnt,
    output reg  [RTU_CHILD_BITS-1:0]      ordcnt,
    output reg                            prochit,
    output reg  [NODE_W-1:0][31:0]        ordt,
    output reg  [NODE_W-1:0][31:0]        ordoff
);
    wire box_c  = box_valid && box_sel;
    wire proc_c = box_c && proc;
    wire tree_c = box_c && ~proc;
    wire ins_c  = tree_c && box_pushable;

    always_ff @(posedge clk) begin
        if (reset) begin
            busy <= 1'b0;
        end else begin
            if (free) begin
                busy <= 1'b0;
            end else if (alloc) begin
                busy <= 1'b1;
            end
            if (alloc) begin
                proc <= alloc_proc;
            end
            if (tree_c) begin
                cnt <= cnt + RTU_CHILD_BITS'(1);
            end else if (alloc) begin
                cnt <= '0;
            end
            if (ins_c) begin
                ordcnt <= ordcnt + RTU_CHILD_BITS'(1);
            end else if (alloc) begin
                ordcnt <= '0;
            end
            if (proc_c) begin
                prochit <= box_hit;
            end
        end
    end

    // The entry decodes its own insertion slot, so the ordering write is one
    // mux per slot. The admit bits are a prefix of ones (the list is sorted
    // ascending and the count mask is a prefix), so the insertion slot is the
    // first zero: a neighbouring-bit decode.
    wire [NODE_W-1:0] ins_le;
    for (genvar oc = 0; oc < NODE_W; ++oc) begin : g_le
        assign ins_le[oc] = (RTU_CHILD_BITS'(oc) < ordcnt) && ins_t_le[oc];
    end

    // a procedural box stores its t in slot 0
    for (genvar oc = 0; oc < NODE_W; ++oc) begin : g_slot
        localparam PREV = (oc != 0) ? (oc - 1) : 0;
        wire ins_here  = ~ins_le[oc] && ((oc == 0) || ins_le[PREV]);
        wire ins_shift = (oc != 0) && ~ins_le[PREV];
        wire set_t   = ((oc == 0) && proc_c) || (ins_c && ins_here);
        wire set_off = ins_c && ins_here;
        wire shift_c = ins_c && ins_shift;
        always_ff @(posedge clk) begin
            if (~reset) begin
                if (set_t) begin
                    ordt[oc] <= box_t_near;
                end else if (shift_c) begin
                    ordt[oc] <= ordt[PREV];
                end
                if (set_off) begin
                    ordoff[oc] <= box_childoff;
                end else if (shift_c) begin
                    ordoff[oc] <= ordoff[PREV];
                end
            end
        end
    end

endmodule

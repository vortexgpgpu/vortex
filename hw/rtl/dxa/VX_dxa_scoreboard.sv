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

// DXA metadata-only scoreboard for OoO rd_ctrl.
// Stores offset, length, and SMEM address per slot (no 512-bit response data).
// Provides free-slot search and indexed metadata lookup for response handling.

`include "VX_define.vh"

module VX_dxa_scoreboard #(
    parameter MAX_OUTSTANDING = 16,
    parameter GMEM_OFF_BITS   = 6,
    parameter SMEM_ADDR_W     = `MEM_ADDR_WIDTH
) (
    input  wire clk,
    input  wire reset,

    // Free-slot search (combinatorial, accounts for same-cycle release).
    output wire                        free_found,
    output wire [`UP(`CLOG2(MAX_OUTSTANDING))-1:0] free_slot,

    // Allocation port.
    input  wire                        alloc_fire,
    input  wire [`UP(`CLOG2(MAX_OUTSTANDING))-1:0] alloc_slot,
    input  wire [GMEM_OFF_BITS-1:0]    alloc_offset,
    input  wire [GMEM_OFF_BITS:0]      alloc_length,
    input  wire [SMEM_ADDR_W-1:0]      alloc_smem_addr,

    // Response lookup (indexed read, combinatorial).
    input  wire [`UP(`CLOG2(MAX_OUTSTANDING))-1:0] rsp_slot,
    output wire                        rsp_slot_busy,
    output wire [GMEM_OFF_BITS-1:0]    rsp_offset,
    output wire [GMEM_OFF_BITS:0]      rsp_length,
    output wire [SMEM_ADDR_W-1:0]      rsp_smem_addr,

    // Release port.
    input  wire                        release_fire,
    input  wire [`UP(`CLOG2(MAX_OUTSTANDING))-1:0] release_slot
);
    localparam SLOT_W = `UP(`CLOG2(MAX_OUTSTANDING));

    reg [MAX_OUTSTANDING-1:0]                   busy_r;
    reg [MAX_OUTSTANDING-1:0][GMEM_OFF_BITS-1:0] offset_r;
    reg [MAX_OUTSTANDING-1:0][GMEM_OFF_BITS:0]   length_r;
    reg [MAX_OUTSTANDING-1:0][SMEM_ADDR_W-1:0]   smem_addr_r;

    // Effective busy: account for same-cycle release.
    wire [MAX_OUTSTANDING-1:0] busy_eff_w;
    for (genvar i = 0; i < MAX_OUTSTANDING; ++i) begin : g_busy_eff
        assign busy_eff_w[i] = busy_r[i] && !(release_fire && (release_slot == SLOT_W'(i)));
    end

    // Free-slot priority search.
    reg                free_found_r;
    reg [SLOT_W-1:0]   free_slot_r;
    always @(*) begin
        free_found_r = 1'b0;
        free_slot_r  = '0;
        for (integer i = 0; i < MAX_OUTSTANDING; ++i) begin
            if (!busy_eff_w[i] && !free_found_r) begin
                free_found_r = 1'b1;
                free_slot_r  = SLOT_W'(i);
            end
        end
    end
    assign free_found = free_found_r;
    assign free_slot  = free_slot_r;

    // Response lookup: pure indexed reads.
    assign rsp_slot_busy = busy_r[rsp_slot];
    assign rsp_offset    = offset_r[rsp_slot];
    assign rsp_length    = length_r[rsp_slot];
    assign rsp_smem_addr = smem_addr_r[rsp_slot];

    // State update.
    always @(posedge clk) begin
        if (reset) begin
            busy_r      <= '0;
            offset_r    <= '0;
            length_r    <= '0;
            smem_addr_r <= '0;
        end else begin
            if (release_fire) begin
                busy_r[release_slot] <= 1'b0;
            end
            if (alloc_fire) begin
                busy_r[alloc_slot]      <= 1'b1;
                offset_r[alloc_slot]    <= alloc_offset;
                length_r[alloc_slot]    <= alloc_length;
                smem_addr_r[alloc_slot] <= alloc_smem_addr;
            end
        end
    end
endmodule

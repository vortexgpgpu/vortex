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

// Walk cache: a small direct-mapped array of last-level page-table pointers.
// A walk indexes it by the VPN bits above the last level; a hit lets the
// walker skip the interior fetches and go straight to the leaf. Entries are
// written as a walk descends into its last level and cleared on flush.
module VX_ptw_cache import VX_tlb_pkg::*; #(
    parameter NUM_ENTRIES = `VX_CFG_PTW_WALK_CACHE_SIZE
) (
    input wire clk,
    input wire reset,

    // Combinational probe (dispatch).
    input  wire [TLB_VPN_WIDTH-1:0]  probe_vpn,
    output wire                      probe_hit,
    output wire [TLB_PPN_WIDTH-1:0]  probe_ppn,

    // Fill (a walk reaching its last level).
    input  wire                      wr_valid,
    input  wire [TLB_VPN_WIDTH-1:0]  wr_vpn,
    input  wire [TLB_PPN_WIDTH-1:0]  wr_ppn,

    input  wire                      flush
);
    localparam TAG_WIDTH = TLB_VPN_WIDTH - TLB_LEVEL_BITS;
    localparam IDX_WIDTH = `CLOG2(NUM_ENTRIES);

    // The tag is the VPN with its last-level index stripped; the low bits of
    // that tag select the array row.
    // Only the VPN above the last level indexes the cache; the low bits (the
    // last-level offset) are the same for every entry a table covers.
    `UNUSED_VAR (probe_vpn[TLB_LEVEL_BITS-1:0])
    `UNUSED_VAR (wr_vpn[TLB_LEVEL_BITS-1:0])

    wire [TAG_WIDTH-1:0] probe_tag = probe_vpn[TLB_VPN_WIDTH-1:TLB_LEVEL_BITS];
    wire [TAG_WIDTH-1:0] wr_tag    = wr_vpn[TLB_VPN_WIDTH-1:TLB_LEVEL_BITS];
    wire [IDX_WIDTH-1:0] probe_idx = probe_tag[IDX_WIDTH-1:0];
    wire [IDX_WIDTH-1:0] wr_idx    = wr_tag[IDX_WIDTH-1:0];

    reg [NUM_ENTRIES-1:0]                 valid_r;
    reg [TAG_WIDTH-1:0]                   tag_r [NUM_ENTRIES];
    reg [TLB_PPN_WIDTH-1:0]               ppn_r [NUM_ENTRIES];

    always @(posedge clk) begin
        if (reset || flush) begin
            valid_r <= '0;
        end else if (wr_valid) begin
            valid_r[wr_idx] <= 1'b1;
        end
    end

    always @(posedge clk) begin
        if (wr_valid) begin
            tag_r[wr_idx] <= wr_tag;
            ppn_r[wr_idx] <= wr_ppn;
        end
    end

    assign probe_hit = valid_r[probe_idx] && (tag_r[probe_idx] == probe_tag);
    assign probe_ppn = ppn_r[probe_idx];

endmodule

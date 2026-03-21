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

// DXA Intra-Row Dedup: 1-entry merge register for CL address dedup.
// Within a row, CL addresses are monotonically non-decreasing, so
// consecutive same-address entries are always adjacent and can be merged
// by OR-ing their byte masks. At row boundaries, addresses may jump
// backwards — dedup only merges consecutive same-address entries.

`include "VX_define.vh"

module VX_dxa_dedup import VX_gpu_pkg::*; #(
    parameter GMEM_LINE_SIZE  = `L1_LINE_SIZE,
    parameter GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE)
) (
    input  wire                        clk,
    input  wire                        reset,

`ifdef PERF_ENABLE
    output wire                        perf_dedup_hit,
`endif

    // Input (from addr_gen).
    input  wire                        in_valid,
    output wire                        in_ready,
    input  wire [GMEM_ADDR_WIDTH-1:0]  in_cl_addr,
    input  wire [GMEM_LINE_SIZE-1:0]   in_byte_mask,
    input  wire                        in_oob,
    input  wire                        in_last,

    // Output (to rd_ctrl).
    output wire                        out_valid,
    input  wire                        out_ready,
    output wire [GMEM_ADDR_WIDTH-1:0]  out_cl_addr,
    output wire [GMEM_LINE_SIZE-1:0]   out_byte_mask,
    output wire                        out_oob,
    output wire                        out_last
);
    // Merge register: holds the current entry being accumulated.
    reg                        mreg_valid_r;
    reg [GMEM_ADDR_WIDTH-1:0]  mreg_addr_r;
    reg [GMEM_LINE_SIZE-1:0]   mreg_mask_r;
    reg                        mreg_oob_r;
    reg                        mreg_last_r;

    // Can we merge the incoming entry with the merge register?
    // Same address AND same oob status (don't merge OOB with non-OOB).
    wire can_merge = mreg_valid_r && (in_cl_addr == mreg_addr_r) && (in_oob == mreg_oob_r);

    // Need flush: mreg has data and incoming can't merge (different address).
    wire need_flush = mreg_valid_r && in_valid && !can_merge;

    // Output drives from merge register.
    // Valid when: flushing (new entry forces mreg out), or mreg has last and no more input.
    assign out_valid     = need_flush || (mreg_valid_r && mreg_last_r && !in_valid);
    assign out_cl_addr   = mreg_addr_r;
    assign out_byte_mask = mreg_mask_r;
    assign out_oob       = mreg_oob_r;
    assign out_last      = mreg_last_r;

    // Accept input when:
    // - mreg empty (load directly)
    // - can merge (OR masks)
    // - flushing and output accepted (flush mreg, load new)
    assign in_ready = !mreg_valid_r || can_merge || (need_flush && out_ready);

    wire in_fire = in_valid && in_ready;

    always @(posedge clk) begin
        if (reset) begin
            mreg_valid_r <= 1'b0;
        end else begin
            if (need_flush && out_ready) begin
                // Flush current mreg, load incoming entry.
                mreg_addr_r  <= in_cl_addr;
                mreg_mask_r  <= in_byte_mask;
                mreg_oob_r   <= in_oob;
                mreg_last_r  <= in_last;
                // mreg_valid stays 1 (replaced, not cleared).
            end else if (can_merge && in_fire) begin
                // Merge: OR masks, propagate last.
                mreg_mask_r <= mreg_mask_r | in_byte_mask;
                mreg_last_r <= mreg_last_r | in_last;
            end else if (!mreg_valid_r && in_valid) begin
                // Load first entry.
                mreg_valid_r <= 1'b1;
                mreg_addr_r  <= in_cl_addr;
                mreg_mask_r  <= in_byte_mask;
                mreg_oob_r   <= in_oob;
                mreg_last_r  <= in_last;
            end else if (mreg_valid_r && mreg_last_r && !in_valid && out_ready) begin
                // Final flush: mreg has last, no more input.
                mreg_valid_r <= 1'b0;
            end
        end
    end

`ifdef PERF_ENABLE
    assign perf_dedup_hit = can_merge && in_fire;
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && out_valid && out_ready) begin
            $write("DXA_PIPE,%0d,DD_OUT,addr=0x%0h,mask=0x%0h,oob=%0d,last=%0d\n",
                $time, mreg_addr_r, mreg_mask_r, mreg_oob_r, mreg_last_r);
        end
    end
`endif

endmodule

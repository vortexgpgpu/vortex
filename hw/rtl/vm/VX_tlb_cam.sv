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

// Fully-associative L1 TLB storage: one FF entry array read in parallel by
// every lane, a single fill/flush write port, and MRU-style replacement.
// Superpage entries mask the low level*TLB_LEVEL_BITS VPN bits on compare
// and splice them back into the PPN on a hit.
module VX_tlb_cam import VX_tlb_pkg::*; #(
    parameter NUM_REQS = 4,
    parameter TLB_SIZE  = 16
) (
    input wire clk,
    input wire reset,

    // Per-lane combinational lookup.
    input  wire [NUM_REQS-1:0][TLB_VPN_WIDTH-1:0] lookup_vpn,
    output wire [NUM_REQS-1:0]                     lookup_hit,
    output wire [NUM_REQS-1:0][TLB_PPN_WIDTH-1:0]  lookup_ppn,
    output wire [NUM_REQS-1:0][TLB_FLAGS_WIDTH-1:0] lookup_flags,
    // Raw (unspliced) page number and level of the matched entry: consumers
    // that re-install the translation elsewhere splice it themselves.
    output wire [NUM_REQS-1:0][TLB_PPN_WIDTH-1:0]  lookup_ppn_raw,
    output wire [NUM_REQS-1:0][TLB_LEVEL_WIDTH-1:0] lookup_level,

    // MRU bump: asserted the cycle a lane consumes its hit.
    input  wire [NUM_REQS-1:0]                     access_hit,

    // Single fill write port (broadcast install).
    input  wire                                     install_valid,
    input  tlb_entry_t                              install_entry,
    output wire                                     install_evict,

    // Flush: invalidate every entry.
    input  wire                                     flush
);
    localparam IDX_W = `CLOG2(TLB_SIZE);

    tlb_entry_t             entries_r [TLB_SIZE];
    reg [TLB_SIZE-1:0]      valid_r;
    reg [TLB_SIZE-1:0]      mru_r;

    // Mask that drops the low level*TLB_LEVEL_BITS VPN bits (superpages).
    function automatic [TLB_VPN_WIDTH-1:0] vpn_mask (input [TLB_LEVEL_WIDTH-1:0] level);
        vpn_mask = {TLB_VPN_WIDTH{1'b1}} << (level * TLB_LEVEL_BITS);
    endfunction

    // ---------------------------------------------------------------------
    // Per-lane lookup
    // ---------------------------------------------------------------------
    wire [NUM_REQS-1:0][IDX_W-1:0] hit_index;

    // Selectable fields packed once so each lane picks the winning entry with a
    // one-hot mux rather than encoding a hit index and re-indexing entries_r.
    localparam SEL_W = TLB_PPN_WIDTH + TLB_FLAGS_WIDTH + TLB_LEVEL_WIDTH;
    wire [TLB_SIZE-1:0][SEL_W-1:0] entry_sel;
    for (genvar i = 0; i < TLB_SIZE; ++i) begin : g_entry_sel
        assign entry_sel[i] = {entries_r[i].ppn, entries_r[i].flags, entries_r[i].level};
    end

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_lookup
        wire [TLB_SIZE-1:0] hit_vec;
        for (genvar i = 0; i < TLB_SIZE; ++i) begin : g_match
            wire [TLB_VPN_WIDTH-1:0] mask = vpn_mask(entries_r[i].level);
            assign hit_vec[i] = valid_r[i]
                && ((entries_r[i].vpn & mask) == (lookup_vpn[l] & mask));
        end

        // Lowest matching index wins: the encoder yields the hit flag, the
        // winning one-hot, and its index (for the MRU bump) without a mux tree.
        wire [TLB_SIZE-1:0] hit_onehot;
        VX_priority_encoder #(
            .N       (TLB_SIZE),
            .REVERSE (0)
        ) hit_enc (
            .data_in    (hit_vec),
            .onehot_out (hit_onehot),
            .index_out  (hit_index[l]),
            .valid_out  (lookup_hit[l])
        );

        // One-hot AND-OR select of the winning entry's fields, off the idx cone.
        wire [SEL_W-1:0] sel;
        VX_onehot_mux #(
            .DATAW (SEL_W),
            .N     (TLB_SIZE)
        ) sel_mux (
            .data_in  (entry_sel),
            .sel_in   (hit_onehot),
            .data_out (sel)
        );
        wire [TLB_PPN_WIDTH-1:0]   sel_ppn   = sel[TLB_LEVEL_WIDTH+TLB_FLAGS_WIDTH +: TLB_PPN_WIDTH];
        wire [TLB_FLAGS_WIDTH-1:0] sel_flags = sel[TLB_LEVEL_WIDTH +: TLB_FLAGS_WIDTH];
        wire [TLB_LEVEL_WIDTH-1:0] sel_level = sel[0 +: TLB_LEVEL_WIDTH];

        wire [TLB_VPN_WIDTH-1:0]   low = lookup_vpn[l] & ~vpn_mask(sel_level);
        // Splice the superpage's intra-page index from the VPN into the PPN.
        assign lookup_ppn[l]     = sel_ppn | TLB_PPN_WIDTH'(low);
        assign lookup_flags[l]   = sel_flags;
        assign lookup_ppn_raw[l] = sel_ppn;
        assign lookup_level[l]   = sel_level;
    end

    // ---------------------------------------------------------------------
    // Victim selection: first invalid, else first non-MRU, else slot 0.
    // ---------------------------------------------------------------------
    wire has_invalid = (| (~valid_r));
    wire has_non_mru = (| (valid_r & ~mru_r));

    reg [IDX_W-1:0] victim;
    always @(*) begin
        victim = '0;
        if (has_invalid) begin
            for (int i = TLB_SIZE-1; i >= 0; --i) begin
                if (!valid_r[i]) begin
                    victim = IDX_W'(i);
                end
            end
        end else if (has_non_mru) begin
            for (int i = TLB_SIZE-1; i >= 0; --i) begin
                if (!mru_r[i]) begin
                    victim = IDX_W'(i);
                end
            end
        end
    end

    // All-valid + all-MRU: clear the MRU bits this install and evict slot 0.
    wire all_mru = ~has_invalid && ~has_non_mru;

    assign install_evict = install_valid && valid_r[victim];

    // ---------------------------------------------------------------------
    // State update
    // ---------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset || flush) begin
            valid_r <= '0;
            mru_r   <= '0;
        end else begin
            // MRU bump on consumed hits.
            for (int l = 0; l < NUM_REQS; ++l) begin
                if (access_hit[l]) begin
                    mru_r[hit_index[l]] <= 1'b1;
                end
            end
            if (install_valid) begin
                if (all_mru) begin
                    mru_r <= '0;
                end
                entries_r[victim] <= install_entry;
                valid_r[victim]   <= 1'b1;
                mru_r[victim]     <= 1'b1;
            end
        end
    end

endmodule

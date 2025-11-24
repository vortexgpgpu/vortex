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

// Author: Yang-gon Kim (yanggon@g.ucla.edu, ORCAS Lab)
// Date: 14/11/2025

// Translation Lookaside Buffer (TLB) Module
// This module implements a fully-associative TLB cache for virtual-to-physical address translation
// Features:
// - 32 entries (configurable via TLB_SIZE parameter)
// - Pseudo-LRU replacement policy using MRU (Most Recently Used) tracking bits
// - Permission checking (Read, Write, Execute)
// - Support for different page sizes (4KB base + superpages)
// - Single-cycle lookup for hits
// - Compatible with both SV32 and SV39 modes

module VX_tlb import VX_gpu_pkg::*; #(
    parameter TLB_SIZE    = `TLB_SIZE,           // Number of TLB entries (default 32)
    parameter XLEN        = `XLEN,               // Address width (32 or 64 bits)
    parameter NUM_LANES   = 1,                   // Number of parallel lanes for multi-lane support

    // Derived parameters based on VM mode
    `ifdef XLEN_64
        parameter VPN_WIDTH   = 27,              // Virtual Page Number width for SV39: 39-12 = 27 bits
        parameter PPN_WIDTH   = 44,              // Physical Page Number width for SV39: 56-12 = 44 bits
    `else
        parameter VPN_WIDTH   = 20,              // Virtual Page Number width for SV32: 32-12 = 20 bits
        parameter PPN_WIDTH   = 22,              // Physical Page Number width for SV32: 34-12 = 22 bits
    `endif
) (
    input  wire                             clk,
    input  wire                             reset,

    // Lookup Interface (from LSU or Fetch unit)
    input  wire                             i_lookup_valid,
    input  wire [NUM_LANES-1:0]             i_lookup_lane_mask,      // Which lanes are active
    input  wire [NUM_LANES-1:0][XLEN-1:0]   i_lookup_vaddr,          // Virtual addresses to translate
    input  wire [1:0]                       i_lookup_access_type,    // 00=LOAD, 01=STORE, 10=FETCH
    output wire                             o_lookup_ready,

    // Lookup Response
    output reg                              o_lookup_hit,            // TLB hit/miss
    output reg  [NUM_LANES-1:0]             o_lookup_hit_mask,       // Per-lane hit status
    output reg  [NUM_LANES-1:0][PPN_WIDTH-1:0] o_lookup_ppn,        // Physical page numbers
    output reg  [NUM_LANES-1:0][7:0]        o_lookup_flags,          // Permission flags per lane
    output reg  [NUM_LANES-1:0]             o_lookup_fault,          // Permission fault per lane
    output reg  [NUM_LANES-1:0][5:0]        o_lookup_size_bits,      // Page size (12=4KB, 21=2MB, 30=1GB)

    // Fill Interface (from Page Table Walker)
    input  wire                             i_fill_valid,
    input  wire [VPN_WIDTH-1:0]             i_fill_vpn,              // Virtual page number to add
    input  wire [PPN_WIDTH-1:0]             i_fill_ppn,              // Physical page number
    input  wire [7:0]                       i_fill_flags,            // PTE flags (DAGUXWRV)
    input  wire [5:0]                       i_fill_size_bits,        // Page size in bits

    // Flush Interface (for SFENCE.VMA)
    input  wire                             i_flush_all,             // Flush entire TLB
    input  wire                             i_flush_valid,           // Flush specific entry
    input  wire [VPN_WIDTH-1:0]             i_flush_vpn              // VPN to flush
);

    // TLB Entry Structure - Optimized for hardware
    typedef struct packed {
        logic                    valid;           // Entry is valid
        logic [VPN_WIDTH-1:0]    vpn;             // Virtual page number
        logic [PPN_WIDTH-1:0]    ppn;             // Physical page number
        logic [7:0]              flags;           // Permission flags (RISC-V PTE format)
        logic [5:0]              size_bits;       // Page size (12 for 4KB, 21 for 2MB, 30 for 1GB)
        logic                    mru;             // Most Recently Used bit for tracking access recency
    } tlb_entry_t;

    // TLB storage array
    tlb_entry_t [TLB_SIZE-1:0] tlb_entries;

    // Internal signals
    logic [TLB_SIZE-1:0]       entry_match [NUM_LANES];
    logic [NUM_LANES-1:0]      lane_hit;
    logic [$clog2(TLB_SIZE)-1:0] hit_index [NUM_LANES];
    logic                      tlb_full;
    logic [$clog2(TLB_SIZE)-1:0] victim_index;
    logic                      evict_needed;

    // Flag bit positions in RISC-V PTE format (for readability)
    localparam FLAG_V = 0;  // Valid : should be same as tlb_entry_t.valid bit. PTW only fills the valid entry. (kind of redundant but useful)
    localparam FLAG_R = 1;  // Read
    localparam FLAG_W = 2;  // Write
    localparam FLAG_X = 3;  // Execute
    localparam FLAG_U = 4;  // User
    localparam FLAG_G = 5;  // Global
    localparam FLAG_A = 6;  // Accessed
    localparam FLAG_D = 7;  // Dirty

    // Count valid entries
    logic [$clog2(TLB_SIZE):0] valid_count;
    always_comb begin
        valid_count = 0;
        for (int i = 0; i < TLB_SIZE; i++) begin
            if (tlb_entries[i].valid) begin
                valid_count = valid_count + 1;
            end
        end
    end

    assign tlb_full = (valid_count == TLB_SIZE);

    // Lookup Logic - Check all entries in parallel for each lane
    always_comb begin
        for (int lane = 0; lane < NUM_LANES; lane++) begin
            entry_match[lane] = '0;
            lane_hit[lane] = 1'b0;
            hit_index[lane] = '0;

            if (i_lookup_lane_mask[lane]) begin
                // Extract VPN from virtual address (considering different page sizes)
                logic [VPN_WIDTH-1:0] incoming_vpn;
                logic [VPN_WIDTH-1:0] vpn_mask;
                logic [5:0] shift_amount;

                // Get the full VPN from the virtual address (bits [XLEN-1:12])
                incoming_vpn = i_lookup_vaddr[lane][XLEN-1:12];

                for (int i = 0; i < TLB_SIZE; i++) begin
                    if (tlb_entries[i].valid) begin
                        // Calculate shift amount based on page size
                        // For 4KB (size_bits=12): shift_amount=0
                        // For 2MB (size_bits=21): shift_amount=9
                        // For 1GB (size_bits=30): shift_amount=18
                        shift_amount = tlb_entries[i].size_bits - 12;

                        // Compare VPNs at the same granularity
                        // The stored VPN is pre-shifted, so we shift the incoming VPN
                        logic [VPN_WIDTH-1:0] incoming_vpn_shifted;
                        incoming_vpn_shifted = incoming_vpn >> shift_amount;

                        if (incoming_vpn_shifted == tlb_entries[i].vpn) begin
                            entry_match[lane][i] = 1'b1;
                            lane_hit[lane] = 1'b1;
                            hit_index[lane] = i[$clog2(TLB_SIZE)-1:0];
                        end
                    end
                end
            end
        end
    end

    // Output generation and permission checking
    always_comb begin
        // Only signal hit when lookup is actually valid
        o_lookup_hit = i_lookup_valid & (|lane_hit);
        o_lookup_hit_mask = i_lookup_valid ? (lane_hit & i_lookup_lane_mask) : '0;

        for (int lane = 0; lane < NUM_LANES; lane++) begin
            o_lookup_ppn[lane] = '0;
            o_lookup_flags[lane] = '0;
            o_lookup_fault[lane] = 1'b0;
            o_lookup_size_bits[lane] = 6'd12; // Default 4KB

            if (i_lookup_valid && lane_hit[lane]) begin
                tlb_entry_t entry = tlb_entries[hit_index[lane]];
                o_lookup_ppn[lane] = entry.ppn;
                o_lookup_flags[lane] = entry.flags;
                o_lookup_size_bits[lane] = entry.size_bits;

                // Permission checking based on access type
                // First check Valid flag - if V=0, always fault regardless of other permissions
                if (~entry.flags[FLAG_V]) begin
                    o_lookup_fault[lane] = 1'b1;
                end else begin
                    // Valid PTE - check R/W/X permissions based on access type
                    case (i_lookup_access_type)
                        2'b00: begin // LOAD - requires Read permission
                            o_lookup_fault[lane] = ~entry.flags[FLAG_R];
                        end
                        2'b01: begin // STORE - requires Write permission
                            o_lookup_fault[lane] = ~entry.flags[FLAG_W];
                        end
                        2'b10: begin // FETCH - requires both Read and Execute permissions
                            o_lookup_fault[lane] = ~(entry.flags[FLAG_R] & entry.flags[FLAG_X]);
                        end
                        default: o_lookup_fault[lane] = 1'b0;
                    endcase
                end
            end
        end
    end

    // MRU Update Logic - Set MRU bit for accessed entries
    // This implements pseudo-LRU: track which entries are Most Recently Used,
    // then evict entries that are NOT most recently used (i.e., evict by LRU using MRU tracking)
    always_ff @(posedge clk) begin
        if (reset) begin
            // Clear all MRU bits on reset
            for (int i = 0; i < TLB_SIZE; i++) begin
                tlb_entries[i].mru <= 1'b0;
            end
        end else if (i_lookup_valid && o_lookup_ready && o_lookup_hit) begin
            // Update MRU bits for hit entries
            for (int lane = 0; lane < NUM_LANES; lane++) begin
                if (lane_hit[lane]) begin
                    tlb_entries[hit_index[lane]].mru <= 1'b1;  // Mark as Most Recently Used

                    // If TLB is full, clear all other MRU bits
                    // This creates a new epoch where only the just-accessed entry is marked as MRU
                    // All other entries become candidates for eviction (LRU)
                    if (tlb_full) begin
                        for (int i = 0; i < TLB_SIZE; i++) begin
                            if (i != hit_index[lane]) begin
                                tlb_entries[i].mru <= 1'b0;
                            end
                        end
                    end
                end
            end
        end
    end

    // Victim Selection for Eviction - Find entry that is NOT most recently used
    // This implements LRU eviction by selecting an entry with MRU=0
    always_comb begin
        victim_index = '0;
        evict_needed = tlb_full;

        if (tlb_full) begin
            // Find first entry with MRU bit = 0 (least recently used)
            // We evict by LRU: select an entry that was NOT most recently used
            for (int i = 0; i < TLB_SIZE; i++) begin
                if (tlb_entries[i].valid && !tlb_entries[i].mru) begin
                    victim_index = i[$clog2(TLB_SIZE)-1:0];
                    break;
                end
            end
        end else begin
            // Find first invalid entry
            for (int i = 0; i < TLB_SIZE; i++) begin
                if (!tlb_entries[i].valid) begin
                    victim_index = i[$clog2(TLB_SIZE)-1:0];
                    break;
                end
            end
        end
    end

    // Fill and Flush Logic
    always_ff @(posedge clk) begin
        if (reset || i_flush_all) begin
            // Clear all entries
            for (int i = 0; i < TLB_SIZE; i++) begin
                tlb_entries[i].valid <= 1'b0;
                tlb_entries[i].vpn <= '0;
                tlb_entries[i].ppn <= '0;
                tlb_entries[i].flags <= '0;
                tlb_entries[i].size_bits <= 6'd12;
                tlb_entries[i].mru <= 1'b0;
            end
        end else begin
            // Handle specific flush
            if (i_flush_valid) begin
                for (int i = 0; i < TLB_SIZE; i++) begin
                    if (tlb_entries[i].valid &&
                        (tlb_entries[i].vpn == i_flush_vpn)) begin
                        tlb_entries[i].valid <= 1'b0;
                    end
                end
            end

            // Handle fill operation
            if (i_fill_valid) begin
                logic [$clog2(TLB_SIZE)-1:0] fill_index;

                // Check if entry already exists (update case)
                logic entry_exists = 1'b0;
                for (int i = 0; i < TLB_SIZE; i++) begin
                    if (tlb_entries[i].valid &&
                        (tlb_entries[i].vpn == i_fill_vpn)) begin
                        fill_index = i[$clog2(TLB_SIZE)-1:0];
                        entry_exists = 1'b1;
                        break;
                    end
                end

                // If not exists, use victim selection (LRU eviction)
                if (!entry_exists) begin
                    fill_index = victim_index;

                    // Clear MRU bits when about to fill last empty slot
                    // This prepares for the next epoch of MRU tracking
                    if (valid_count == (TLB_SIZE - 1)) begin
                        for (int i = 0; i < TLB_SIZE; i++) begin
                            tlb_entries[i].mru <= 1'b0;
                        end
                    end
                end

                // Fill the entry - only store flags once!
                tlb_entries[fill_index].valid <= 1'b1;
                tlb_entries[fill_index].vpn <= i_fill_vpn;
                tlb_entries[fill_index].ppn <= i_fill_ppn;
                tlb_entries[fill_index].flags <= i_fill_flags;
                tlb_entries[fill_index].size_bits <= i_fill_size_bits;
                tlb_entries[fill_index].mru <= 1'b1;  // Mark new entry as Most Recently Used
            end
        end
    end

    // Ready signal - TLB can always accept lookups
    assign o_lookup_ready = 1'b1;

    // Assertions for verification
    `ifdef SIMULATION
        // Check that only one entry matches per lookup
        always_ff @(posedge clk) begin
            if (i_lookup_valid && !reset) begin
                for (int lane = 0; lane < NUM_LANES; lane++) begin
                    automatic int match_count = 0;
                    for (int i = 0; i < TLB_SIZE; i++) begin
                        if (entry_match[lane][i]) match_count++;
                    end
                    assert(match_count <= 1) else
                        $error("Multiple TLB entries match for lane %0d", lane);
                end
            end
        end

        // Check that LRU eviction policy is working correctly
        // We evict by LRU using MRU tracking - verify we never evict an entry with MRU=1
        always_ff @(posedge clk) begin
            if (i_fill_valid && evict_needed && !reset) begin
                assert(!tlb_entries[victim_index].mru) else
                    $error("Evicting entry with MRU=1 (violates LRU policy)");
            end
        end

        // Verify flags are valid PTE format
        always_ff @(posedge clk) begin
            if (i_fill_valid && !reset) begin
                // Check that if W=1, then R must also be 1 (RISC-V requirement)
                if (i_fill_flags[FLAG_W]) begin
                    assert(i_fill_flags[FLAG_R]) else
                        $error("Invalid PTE: W=1 but R=0");
                end
            end
        end
    `endif

endmodule
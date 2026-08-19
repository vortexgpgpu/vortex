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

// One bank of the per-core TLB: a fully-associative CAM with MRU
// replacement and a lookup/miss/replay state machine. Hits pass through in
// the same cycle; a miss parks the request, sends one walk request to the
// shared PTW and replays the request once the translation has been filled.
// Superpage entries match on the VPN bits above their page level.
module VX_mmu_tlb_bank import VX_gpu_pkg::*; #(
    parameter NUM_ENTRIES   = 32,
    parameter DATA_SIZE     = DCACHE_WORD_SIZE,
    parameter ADDR_WIDTH    = DCACHE_ADDR_WIDTH,
    parameter TAG_WIDTH_IN  = DCACHE_TAG_WIDTH_BASE,
    parameter SOURCE_BITS   = 1,
    parameter ATTR_WIDTH    = MEM_ATTR_WIDTH,
    parameter DATA_WIDTH    = DATA_SIZE * 8,
    parameter TAG_WIDTH_OUT = TAG_WIDTH_IN + SOURCE_BITS,
    parameter REQ_DATAW_IN  = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + ATTR_WIDTH + TAG_WIDTH_IN,
    parameter REQ_DATAW_OUT = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + ATTR_WIDTH + TAG_WIDTH_OUT
) (
    input wire clk,
    input wire reset,
    input wire flush,

    // arbitrated request stream in, with the originating lane
    input  wire                     req_valid,
    input  wire [REQ_DATAW_IN-1:0]  req_data,
    input  wire [SOURCE_BITS-1:0]   req_sel,
    output wire                     req_ready,

    // translated request stream out, lane folded into the tag
    output wire                     out_valid,
    output wire [REQ_DATAW_OUT-1:0] out_data,
    input  wire                     out_ready,

    // page-table walker
    output wire                          miss_valid,
    input  wire                          miss_ready,
    output wire [VM_VPN_WIDTH-1:0]       miss_vpn,

    input  wire                          fill_valid,
    output wire                          fill_ready,
    input  wire [VM_PPN_WIDTH-1:0]       fill_ppn,
    input  wire [VM_LEVEL_BITS-1:0]      fill_level,
    input  wire [VM_PTE_FLAGS_WIDTH-1:0] fill_flags,
    input  wire                          fill_fault,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_tlb_reads,
    output wire [PERF_CTR_BITS-1:0] perf_tlb_hits,
    output wire [PERF_CTR_BITS-1:0] perf_tlb_misses,
    output wire [PERF_CTR_BITS-1:0] perf_tlb_evictions
`else
    output wire perf_placeholder
`endif
);
    `STATIC_ASSERT(`IS_POW2(NUM_ENTRIES), ("NUM_ENTRIES must be a power of 2"))

    localparam INDEX_BITS       = `LOG2UP(NUM_ENTRIES);
    localparam PAGE_OFFSET_BITS = VM_PAGE_OFFSET_BITS - `CLOG2(DATA_SIZE);
    localparam ADDR_LSB_IN      = TAG_WIDTH_IN + ATTR_WIDTH + DATA_SIZE + DATA_WIDTH;

    typedef struct packed {
        logic                          valid;
        logic                          mru;
        logic [VM_LEVEL_BITS-1:0]      level;
        logic [VM_VPN_WIDTH-1:0]       vpn;
        logic [VM_PPN_WIDTH-1:0]       ppn;
        logic [VM_PTE_FLAGS_WIDTH-1:0] flags;
    } tlb_entry_t;

    tlb_entry_t entries [NUM_ENTRIES];

    typedef enum logic [1:0] {
        TLB_READY    = 2'd0,
        TLB_PTW_WAIT = 2'd1,
        TLB_REPLAY   = 2'd2
    } tlb_state_t;

    tlb_state_t            state;
    reg [REQ_DATAW_IN-1:0] miss_buffer;
    reg [SOURCE_BITS-1:0]  miss_sel;
    reg                    miss_sent;
    reg                    replay_identity;
    reg [INDEX_BITS-1:0]   victim_index;

    // -------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------

    wire is_replay = (state == TLB_REPLAY);
    wire [REQ_DATAW_IN-1:0] lookup_data = is_replay ? miss_buffer : req_data;
    wire [SOURCE_BITS-1:0]  lookup_sel  = is_replay ? miss_sel : req_sel;
    wire [ADDR_WIDTH-1:0]   lookup_addr = lookup_data[ADDR_LSB_IN +: ADDR_WIDTH];
    wire [VM_VPN_WIDTH-1:0] lookup_vpn  = lookup_addr[PAGE_OFFSET_BITS +: VM_VPN_WIDTH];

    // VPN bits below a superpage's level are part of its page offset.
    function automatic logic [VM_VPN_WIDTH-1:0] vpn_mask(input logic [VM_LEVEL_BITS-1:0] level);
        return ~((VM_VPN_WIDTH'(1) << (level * VM_VPN_LEVEL_BITS)) - VM_VPN_WIDTH'(1));
    endfunction

    wire [NUM_ENTRIES-1:0] cam_hit;
    for (genvar i = 0; i < NUM_ENTRIES; ++i) begin : g_cam
        wire [VM_VPN_WIDTH-1:0] mask = vpn_mask(entries[i].level);
        assign cam_hit[i] = entries[i].valid && ((entries[i].vpn & mask) == (lookup_vpn & mask));
    end

    wire                  tlb_hit;
    wire [INDEX_BITS-1:0] hit_index;

    VX_priority_encoder #(
        .N (NUM_ENTRIES)
    ) hit_enc (
        .data_in   (cam_hit),
        .index_out (hit_index),
        .valid_out (tlb_hit),
        `UNUSED_PIN (onehot_out)
    );

    // -------------------------------------------------------------------------
    // Victim selection: a free entry first, otherwise the first non-MRU one
    // -------------------------------------------------------------------------

    wire [NUM_ENTRIES-1:0] entry_free, entry_not_mru;
    for (genvar i = 0; i < NUM_ENTRIES; ++i) begin : g_victim
        assign entry_free[i]    = ~entries[i].valid;
        assign entry_not_mru[i] = ~entries[i].mru;
    end

    wire                  free_valid;
    wire [INDEX_BITS-1:0] free_index;
    wire [INDEX_BITS-1:0] not_mru_index;

    VX_priority_encoder #(
        .N (NUM_ENTRIES)
    ) free_enc (
        .data_in   (entry_free),
        .index_out (free_index),
        .valid_out (free_valid),
        `UNUSED_PIN (onehot_out)
    );

    VX_priority_encoder #(
        .N (NUM_ENTRIES)
    ) not_mru_enc (
        .data_in   (entry_not_mru),
        .index_out (not_mru_index),
        `UNUSED_PIN (valid_out),
        `UNUSED_PIN (onehot_out)
    );

    wire [INDEX_BITS-1:0] victim_candidate = free_valid ? free_index : not_mru_index;
    wire all_mru = ~(|entry_not_mru);

    // -------------------------------------------------------------------------
    // Translation
    // -------------------------------------------------------------------------

    wire [VM_PPN_WIDTH-1:0]   hit_ppn   = entries[hit_index].ppn;
    wire [VM_LEVEL_BITS-1:0]  hit_level = entries[hit_index].level;

    // A level-L entry keeps the low L*VPN_LEVEL_BITS VPN bits as offset.
    wire [ADDR_WIDTH-1:0] hit_page_mask = ~((ADDR_WIDTH'(1) << (PAGE_OFFSET_BITS + hit_level * VM_VPN_LEVEL_BITS)) - ADDR_WIDTH'(1));
    wire [ADDR_WIDTH-1:0] hit_page_addr = ADDR_WIDTH'({hit_ppn, {PAGE_OFFSET_BITS{1'b0}}});
    wire [ADDR_WIDTH-1:0] cam_translated_addr = (hit_page_addr & hit_page_mask) | (lookup_addr & ~hit_page_mask);

    wire [ADDR_WIDTH-1:0] translated_addr = (is_replay && replay_identity) ? lookup_addr : cam_translated_addr;

    wire [TAG_WIDTH_OUT-1:0] lookup_tag_out;
    VX_bits_insert #(
        .N   (TAG_WIDTH_IN),
        .S   (SOURCE_BITS),
        .POS (0)
    ) tag_insert (
        .data_in  (lookup_data[TAG_WIDTH_IN-1:0]),
        .ins_in   (lookup_sel),
        .data_out (lookup_tag_out)
    );

    assign out_data = {
        lookup_data[REQ_DATAW_IN-1],
        translated_addr,
        lookup_data[ADDR_LSB_IN-1:TAG_WIDTH_IN],
        lookup_tag_out
    };

    // -------------------------------------------------------------------------
    // Control
    // -------------------------------------------------------------------------

    wire req_fire  = req_valid && req_ready;
    wire out_fire  = out_valid && out_ready;
    wire miss_fire = miss_valid && miss_ready;
    wire fill_fire = fill_valid && fill_ready;
    wire replay_hit = tlb_hit || replay_identity;

    assign req_ready  = (state == TLB_READY) && (out_ready || !tlb_hit);
    assign out_valid  = ((state == TLB_READY) && req_valid && tlb_hit)
                     || (is_replay && replay_hit);

    assign miss_valid = (state == TLB_PTW_WAIT) && !miss_sent;
    assign miss_vpn   = miss_buffer[ADDR_LSB_IN + PAGE_OFFSET_BITS +: VM_VPN_WIDTH];
    assign fill_ready = (state == TLB_PTW_WAIT) && miss_sent;

    wire install = fill_fire && !fill_fault && !flush;

    always @(posedge clk) begin
        if (reset) begin
            state           <= TLB_READY;
            miss_sent       <= 1'b0;
            replay_identity <= 1'b0;
        end else begin
            case (state)
            TLB_READY: begin
                if (req_fire && !tlb_hit) begin
                    miss_buffer     <= req_data;
                    miss_sel        <= req_sel;
                    victim_index    <= victim_candidate;
                    miss_sent       <= 1'b0;
                    replay_identity <= 1'b0;
                    state           <= TLB_PTW_WAIT;
                end
            end
            TLB_PTW_WAIT: begin
                if (miss_fire) begin
                    miss_sent <= 1'b1;
                end
                if (fill_fire) begin
                    replay_identity <= fill_fault;
                    state           <= TLB_REPLAY;
                end
            end
            TLB_REPLAY: begin
                if (out_fire) begin
                    state <= TLB_READY;
                end else if (!replay_hit) begin
                    // the filled entry was flushed before the replay; walk again
                    miss_sent <= 1'b0;
                    state     <= TLB_PTW_WAIT;
                end
            end
            default:;
            endcase
        end
    end

    // entry storage
    always @(posedge clk) begin
        if (reset || flush) begin
            for (integer i = 0; i < NUM_ENTRIES; ++i) begin
                entries[i].valid <= 1'b0;
                entries[i].mru   <= 1'b0;
            end
        end else begin
            if ((state == TLB_READY) && req_fire && tlb_hit) begin
                entries[hit_index].mru <= 1'b1;
                if (all_mru) begin
                    for (integer i = 0; i < NUM_ENTRIES; ++i) begin
                        if (INDEX_BITS'(i) != hit_index) entries[i].mru <= 1'b0;
                    end
                end
            end
            if (install) begin
                entries[victim_index].valid <= 1'b1;
                entries[victim_index].mru   <= 1'b1;
                entries[victim_index].level <= fill_level;
                entries[victim_index].vpn   <= miss_vpn;
                entries[victim_index].ppn   <= fill_ppn;
                entries[victim_index].flags <= fill_flags;
                if (all_mru) begin
                    for (integer i = 0; i < NUM_ENTRIES; ++i) begin
                        if (INDEX_BITS'(i) != victim_index) entries[i].mru <= 1'b0;
                    end
                end
            end
        end
    end

`ifdef DBG_TRACE_MMU
    always @(posedge clk) begin
        if ((state == TLB_READY) && req_fire) begin
            `TRACE(2, ("%t: tlb-lookup: vpn=0x%0h, hit=%b, lane=%0d\n", $time, lookup_vpn, tlb_hit, req_sel))
        end
        if (fill_fire) begin
            `TRACE(2, ("%t: tlb-fill: vpn=0x%0h, ppn=0x%0h, level=%0d, fault=%b, victim=%0d, install=%b, flush=%b\n", $time, miss_vpn, fill_ppn, fill_level, fill_fault, victim_index, install, flush))
        end
        if (is_replay && !replay_hit) begin
            `TRACE(2, ("%t: tlb-replay-miss: vpn=0x%0h, v0=%b, e0vpn=0x%0h, e0lvl=%0d, hit=%b\n", $time, lookup_vpn, entries[0].valid, entries[0].vpn, entries[0].level, tlb_hit))
        end
        if (is_replay && out_fire) begin
            `TRACE(2, ("%t: tlb-replay: vpn=0x%0h, paddr=0x%0h\n", $time, lookup_vpn, translated_addr))
        end
    end
`endif

    // -------------------------------------------------------------------------
    // Performance counters
    // -------------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_reads_r, perf_hits_r, perf_misses_r, perf_evictions_r;

    always @(posedge clk) begin
        if (reset) begin
            perf_reads_r     <= '0;
            perf_hits_r      <= '0;
            perf_misses_r    <= '0;
            perf_evictions_r <= '0;
        end else begin
            if ((state == TLB_READY) && req_fire) begin
                perf_reads_r <= perf_reads_r + PERF_CTR_BITS'(1);
                if (tlb_hit) perf_hits_r <= perf_hits_r + PERF_CTR_BITS'(1);
            end
            if (miss_fire) begin
                perf_misses_r <= perf_misses_r + PERF_CTR_BITS'(1);
            end
            if (install && entries[victim_index].valid) begin
                perf_evictions_r <= perf_evictions_r + PERF_CTR_BITS'(1);
            end
        end
    end

    assign perf_tlb_reads     = perf_reads_r;
    assign perf_tlb_hits      = perf_hits_r;
    assign perf_tlb_misses    = perf_misses_r;
    assign perf_tlb_evictions = perf_evictions_r;
`else
    assign perf_placeholder = 1'b0;
`endif

endmodule

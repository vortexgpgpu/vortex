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

// DXA SMEM Writer (Phase 4b — OOO direct drain).
//
// Receives CLs directly from gmem_req on the `sw_*` channel, no rsp_buf
// in the middle. The pend slot is filled asynchronously by whatever rsp
// (real or OOB-synthetic) gmem_req presents. The CL marked `last` is
// special — it must drain LAST because its bus packet carries
// notify_smem_done; we hold it in `defer_*_r` until all other CLs have
// released, then promote it to pend.
//
// Drain itself stays the same as Phase 1: pend → barrel-shift → fb_data_r
// → SMEM_WORD beats. 1 SMEM-word/cycle steady state.

`include "VX_define.vh"

module VX_dxa_smem_wr import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter CL_SIZE         = `L1_LINE_SIZE,
    parameter SMEM_WORD_SIZE  = DXA_LMEM_WORD_SIZE,
    parameter SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_W,
    parameter GMEM_DATAW      = CL_SIZE * 8
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire                        transfer_active,
    input  wire                        transfer_start,

    // cfill (stable during transfer).
    input  wire [31:0]                 cfill,

    // Metadata for SMEM bus tag + completion attr.
    input  wire [NC_WIDTH-1:0]         active_core_id,
    input  wire [BAR_ADDR_W-1:0]       active_bar_addr,
    input  wire                        active_notify_smem_done,

    // Direct drain channel (from gmem_req).
    input  wire                        sw_valid,
    output wire                        sw_ready,
    input  wire [TAG_W-1:0]            sw_tag,
    input  wire [GMEM_DATAW-1:0]       sw_data,
    input  wire [`MEM_ADDR_WIDTH-1:0]  sw_smem_byte_addr,
    input  wire [CL_OFF_BITS-1:0]      sw_byte_offset,
    input  wire [CL_OFF_BITS:0]        sw_valid_length,
    input  wire                        sw_oob,
    input  wire                        sw_last,
    input  wire [SEQ_W-1:0]            sw_outstanding,

    // Resource release to gmem_req (per-tag, OOO).
    output wire                        release_en,
    output wire [TAG_W-1:0]            release_tag,

    // SMEM bus interface (writes only).
    VX_mem_bus_if.master               smem_bus_if,

    // Completion.
    output wire                        transfer_done,
    output wire [31:0]                 wr_done_count,
    output wire                        smem_req_fire,

    // Multicast (always available; active when is_multicast is set).
    input  wire                        is_multicast,
    input  wire [`NUM_WARPS-1:0]       cta_mask,
    input  wire [31:0]                 smem_stride

`ifdef DXA_OOO_DRAIN_ENABLE
    // OoO transfer_done driven by (addr_gen_done && pending_empty)
    // because lat_last_r is no longer guaranteed to be the last-drained CL.
    ,
    input  wire                        pending_empty,
    input  wire                        addr_gen_done
`endif

`ifdef PERF_ENABLE
    ,
    output wire [31:0]                 perf_lmem_writes
`endif
);
    localparam CL_OFF_BITS  = `CLOG2(CL_SIZE);
    localparam SMEM_OFF_W   = `CLOG2(SMEM_WORD_SIZE);
    localparam SMEM_DATAW   = SMEM_WORD_SIZE * 8;
    localparam TAG_W        = `CLOG2(MAX_OUTSTANDING);
    localparam SEQ_W        = `CLOG2(MAX_OUTSTANDING + 1);
    localparam FILL_CAP     = CL_SIZE + SMEM_WORD_SIZE;
    localparam FILL_W       = `CLOG2(FILL_CAP + 1);

    localparam ENGINE_VALUE_W = DXA_LMEM_ENGINE_TAG_W - UUID_WIDTH;
    localparam SMEM_TAG_VALUE_W = DXA_LMEM_TAG_W - UUID_WIDTH;

    // ════════════════════════════════════════════════════════════════════
    // Cfill replication (for OOB CLs)
    // ════════════════════════════════════════════════════════════════════
    wire [GMEM_DATAW-1:0] cfill_replicated;
    for (genvar i = 0; i < CL_SIZE / 4; ++i) begin : g_cfill
        assign cfill_replicated[i*32 +: 32] = cfill;
    end

    // ════════════════════════════════════════════════════════════════════
    // Pending slot (1-deep skid from sw channel).
    // ════════════════════════════════════════════════════════════════════
    reg                       pend_valid_r;
    reg [TAG_W-1:0]           pend_tag_r;
    reg [`MEM_ADDR_WIDTH-1:0] pend_smem_byte_addr_r;
    reg [CL_OFF_BITS-1:0]     pend_byte_offset_r;
    reg [CL_OFF_BITS:0]       pend_valid_length_r;
    reg                       pend_last_r;
    reg [GMEM_DATAW-1:0]      pend_data_r;

    // ════════════════════════════════════════════════════════════════════
    // Deferred-last slot — holds the CL marked `last` while other CLs
    // drain ahead of it. Promoted to pend when only this CL remains.
    // ════════════════════════════════════════════════════════════════════
    reg                       defer_valid_r;
    reg [TAG_W-1:0]           defer_tag_r;
    reg [`MEM_ADDR_WIDTH-1:0] defer_smem_byte_addr_r;
    reg [CL_OFF_BITS-1:0]     defer_byte_offset_r;
    reg [CL_OFF_BITS:0]       defer_valid_length_r;
    reg [GMEM_DATAW-1:0]      defer_data_r;

    // ════════════════════════════════════════════════════════════════════
    // Drain fill buffer
    // ════════════════════════════════════════════════════════════════════
    reg                        fb_active_r;
    reg [TAG_W-1:0]            fb_tag_r;
    reg                        fb_last_r;
    reg [FILL_CAP*8-1:0]       fb_data_r /*verilator split_var*/;
    reg [FILL_W-1:0]           fb_level_r;
    reg [SMEM_ADDR_WIDTH-1:0]  fb_word_addr_r;
    reg [SMEM_OFF_W-1:0]       fb_byte_offset_r;
    reg [SMEM_ADDR_WIDTH-1:0]  fb_start_word_r;

    // ════════════════════════════════════════════════════════════════════
    // Drain-side combinational
    // ════════════════════════════════════════════════════════════════════
    wire has_full_word    = (fb_level_r >= FILL_W'(SMEM_WORD_SIZE));
    wire has_last_partial = !has_full_word && (fb_level_r > 0);
    wire drain_valid      = fb_active_r && (has_full_word || has_last_partial);
    wire drain_will_empty = (fb_level_r <= FILL_W'(SMEM_WORD_SIZE));

    wire smem_wr_ready_internal;
    wire drain_fire = drain_valid && smem_wr_ready_internal;
    wire drain_emptying_now = drain_fire && drain_will_empty;
    wire fb_will_be_empty   = drain_emptying_now || ~fb_active_r;

    // SMEM word data and byteen from fill buffer.
    wire [SMEM_DATAW-1:0] fb_word_data = fb_data_r[SMEM_DATAW-1:0];
    wire is_first_word = (fb_word_addr_r == fb_start_word_r);

    wire [SMEM_WORD_SIZE-1:0] fb_word_byteen;
    for (genvar i = 0; i < SMEM_WORD_SIZE; ++i) begin : g_byteen
        wire byte_has_data = (FILL_W'(i) < fb_level_r);
        if (i < SMEM_WORD_SIZE - 1) begin : g_with_offset
            wire byte_is_offset = is_first_word && (SMEM_OFF_W'(i) < fb_byte_offset_r);
            assign fb_word_byteen[i] = byte_has_data && !byte_is_offset;
        end else begin : g_no_offset
            assign fb_word_byteen[i] = byte_has_data;
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // sw-channel accept + load scheduling
    // ════════════════════════════════════════════════════════════════════
    //
    // The arriving sw CL takes one of three paths:
    //   (a) Bypass-load into fb_data_r directly (pend is empty AND fb
    //       is empty/emptying).
    //   (b) Capture into pend (pend free, but fb is currently busy or
    //       we'd rather drain a previously-pended CL first).
    //   (c) Capture into defer_*_r (it's the `last` CL and we still have
    //       other CLs outstanding to drain first).
    //
    // Promotion from defer_*_r to pend (or fb directly) fires when the
    // last drain leaves only the deferred CL outstanding.

    wire sw_is_last_defer = sw_valid && sw_last && (sw_outstanding > SEQ_W'(1));
    wire can_defer        = sw_is_last_defer && ~defer_valid_r;

    wire can_capture_pend = ~pend_valid_r;
    wire can_bypass_load  = ~pend_valid_r && fb_will_be_empty;

    // sw_ready conditions:
    //   - If sw is the last-to-defer: defer is empty → accept.
    //   - Else: pend can capture → accept.
    assign sw_ready = sw_is_last_defer ? can_defer : can_capture_pend;

    wire sw_accept       = sw_valid && sw_ready;
    wire sw_defer_path   = sw_accept && sw_is_last_defer;
    wire sw_bypass_path  = sw_accept && ~sw_is_last_defer && can_bypass_load;
    wire sw_pend_path    = sw_accept && ~sw_is_last_defer && ~can_bypass_load;

    // Deferred-CL promotion to fb: fires when fb is emptying and we are
    // about to be down to 1 outstanding (= just the deferred).
    // Equivalent: sw_outstanding == 2 && drain_emptying_now, or
    // sw_outstanding == 1 with fb idle (no other CL in flight).
    wire promote_defer = defer_valid_r && fb_will_be_empty && ~pend_valid_r
                      && ((sw_outstanding == SEQ_W'(1))
                          || (sw_outstanding == SEQ_W'(2) && drain_emptying_now));

    // pend → fb load (existing pend has data ready, fb emptying).
    wire load_pend_to_fb = pend_valid_r && fb_will_be_empty;

    // ════════════════════════════════════════════════════════════════════
    // Data + metadata muxing for the cycle's fb-load source (if any).
    // ════════════════════════════════════════════════════════════════════
    wire                       use_sw_for_fb    = sw_bypass_path;
    wire                       use_pend_for_fb  = load_pend_to_fb && ~use_sw_for_fb;
    wire                       use_defer_for_fb = promote_defer  && ~use_sw_for_fb && ~use_pend_for_fb;
    wire                       fb_load_now      = use_sw_for_fb || use_pend_for_fb || use_defer_for_fb;

    wire [GMEM_DATAW-1:0]      fb_load_data;
    wire [`MEM_ADDR_WIDTH-1:0] fb_load_smem_byte_addr;
    wire [CL_OFF_BITS-1:0]     fb_load_byte_offset;
    wire [CL_OFF_BITS:0]       fb_load_valid_length;
    wire [TAG_W-1:0]           fb_load_tag;
    wire                       fb_load_last;
    wire                       fb_load_oob;

    assign fb_load_data           = use_sw_for_fb    ? (sw_oob ? cfill_replicated : sw_data)
                                  : use_pend_for_fb  ? pend_data_r
                                                     : defer_data_r;
    assign fb_load_smem_byte_addr = use_sw_for_fb    ? sw_smem_byte_addr
                                  : use_pend_for_fb  ? pend_smem_byte_addr_r
                                                     : defer_smem_byte_addr_r;
    assign fb_load_byte_offset    = use_sw_for_fb    ? sw_byte_offset
                                  : use_pend_for_fb  ? pend_byte_offset_r
                                                     : defer_byte_offset_r;
    assign fb_load_valid_length   = use_sw_for_fb    ? sw_valid_length
                                  : use_pend_for_fb  ? pend_valid_length_r
                                                     : defer_valid_length_r;
    assign fb_load_tag            = use_sw_for_fb    ? sw_tag
                                  : use_pend_for_fb  ? pend_tag_r
                                                     : defer_tag_r;
    assign fb_load_last           = use_sw_for_fb    ? sw_last
                                  : use_pend_for_fb  ? pend_last_r
                                                     : 1'b1;  // promoted defer is the last
    assign fb_load_oob            = use_sw_for_fb    ? sw_oob
                                  : 1'b0;  // pend/defer paths already hold (cfill-or-data) in their data reg

    wire [GMEM_DATAW-1:0] compressed_data = (fb_load_valid_length != 0)
        ? (fb_load_data >> {fb_load_byte_offset, 3'b000})
        : '0;

    wire [SMEM_OFF_W-1:0]      new_smem_byte_off = fb_load_smem_byte_addr[SMEM_OFF_W-1:0];
    wire [SMEM_ADDR_WIDTH-1:0] new_start_word    = SMEM_ADDR_WIDTH'(fb_load_smem_byte_addr >> SMEM_OFF_W);
    wire [FILL_W-1:0]          new_fill_level    = FILL_W'(new_smem_byte_off) + FILL_W'(fb_load_valid_length);
    wire [FILL_W+2:0]          new_bit_offset    = {FILL_W'(new_smem_byte_off), 3'b000};

    // ════════════════════════════════════════════════════════════════════
    // Sequential update
    // ════════════════════════════════════════════════════════════════════
    always @(posedge clk) begin
        if (reset || transfer_start) begin
            pend_valid_r     <= 1'b0;
            defer_valid_r    <= 1'b0;
            fb_active_r      <= 1'b0;
            fb_data_r        <= '0;
            fb_level_r       <= '0;
            fb_word_addr_r   <= '0;
            fb_byte_offset_r <= '0;
            fb_start_word_r  <= '0;
            fb_tag_r         <= '0;
            fb_last_r        <= 1'b0;
        end else begin
            // ── Drain advance (mid-CL beat shift) ──
            if (drain_fire && ~drain_will_empty) begin
                fb_data_r      <= fb_data_r >> SMEM_DATAW;
                fb_level_r     <= fb_level_r - FILL_W'(SMEM_WORD_SIZE);
                fb_word_addr_r <= fb_word_addr_r + SMEM_ADDR_WIDTH'(1);
            end

            // ── Load fill buffer with the next CL ──
            if (fb_load_now) begin
                fb_data_r        <= (FILL_CAP*8)'(compressed_data) << new_bit_offset;
                fb_level_r       <= new_fill_level;
                fb_word_addr_r   <= new_start_word;
                fb_byte_offset_r <= new_smem_byte_off;
                fb_start_word_r  <= new_start_word;
                fb_tag_r         <= fb_load_tag;
                fb_last_r        <= fb_load_last;
                fb_active_r      <= 1'b1;
            end else if (drain_emptying_now) begin
                fb_data_r   <= '0;
                fb_level_r  <= '0;
                fb_active_r <= 1'b0;
            end

            // ── pend slot management ──
            if (sw_pend_path) begin
                pend_tag_r            <= sw_tag;
                pend_smem_byte_addr_r <= sw_smem_byte_addr;
                pend_byte_offset_r    <= sw_byte_offset;
                pend_valid_length_r   <= sw_valid_length;
                pend_last_r           <= sw_last;
                pend_data_r           <= sw_oob ? cfill_replicated : sw_data;
                pend_valid_r          <= 1'b1;
            end else if (use_pend_for_fb) begin
                pend_valid_r <= 1'b0;
            end

            // ── defer slot management ──
            if (sw_defer_path) begin
                defer_tag_r            <= sw_tag;
                defer_smem_byte_addr_r <= sw_smem_byte_addr;
                defer_byte_offset_r    <= sw_byte_offset;
                defer_valid_length_r   <= sw_valid_length;
                defer_data_r           <= sw_oob ? cfill_replicated : sw_data;
                defer_valid_r          <= 1'b1;
            end else if (use_defer_for_fb) begin
                defer_valid_r <= 1'b0;
            end
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Resource release (per-tag, OOO)
    // ════════════════════════════════════════════════════════════════════
    wire drain_complete = drain_fire && drain_will_empty;

    assign release_en  = drain_complete;
    assign release_tag = fb_tag_r;

    // ════════════════════════════════════════════════════════════════════
    // SMEM write output (with optional multicast replay)
    // ════════════════════════════════════════════════════════════════════
    wire                       smem_wr_valid;
    wire [SMEM_ADDR_WIDTH-1:0] smem_wr_addr;
    wire                       smem_wr_last_pkt;

    localparam MC_NW_BITS = `CLOG2(`NUM_WARPS);
    wire [SMEM_ADDR_WIDTH-1:0] smem_stride_words = SMEM_ADDR_WIDTH'(smem_stride >> SMEM_OFF_W);

    reg [`NUM_WARPS-1:0] replay_remaining_r;
    wire [MC_NW_BITS-1:0] replay_next_idx;
    wire replay_has_remaining;

    // Combinational reload: when replay_remaining_r=0 at a new word boundary,
    // present cta_mask to the PE this cycle so the first beat of the new
    // word fires without a 1-cycle reload gap.
    wire reload_now = is_multicast && drain_valid && (replay_remaining_r == '0);
    wire [`NUM_WARPS-1:0] replay_remaining_use = reload_now ? cta_mask : replay_remaining_r;

    VX_priority_encoder #(
        .N (`NUM_WARPS)
    ) replay_pe (
        .data_in   (replay_remaining_use),
        .index_out (replay_next_idx),
        .valid_out (replay_has_remaining),
        `UNUSED_PIN (onehot_out)
    );

    wire [SMEM_ADDR_WIDTH-1:0] replay_addr = fb_word_addr_r
        + SMEM_ADDR_WIDTH'(replay_next_idx) * smem_stride_words;
    wire replay_is_last = replay_has_remaining
        && (replay_remaining_use == (`NUM_WARPS'(1) << replay_next_idx));

    wire mc_write_valid = transfer_active && drain_valid
                       && (!is_multicast || replay_has_remaining);
    wire mc_write_fire = mc_write_valid && smem_bus_if.req_ready;
    assign smem_wr_ready_internal = is_multicast
        ? (smem_bus_if.req_ready && (!replay_has_remaining || replay_is_last))
        : smem_bus_if.req_ready;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            replay_remaining_r <= '0;
        end else if (transfer_active && is_multicast) begin
            // Single update: reload from cta_mask on demand, then clear the
            // bit corresponding to the beat that fires this cycle (if any).
            if (mc_write_fire && replay_has_remaining) begin
                replay_remaining_r <= replay_remaining_use & ~(`NUM_WARPS'(1) << replay_next_idx);
            end else if (reload_now) begin
                replay_remaining_r <= cta_mask;
            end
        end
    end

    assign smem_wr_valid   = mc_write_valid;
    assign smem_wr_addr    = is_multicast ? replay_addr : fb_word_addr_r;
    assign smem_req_fire   = mc_write_fire;

    wire is_last_drain = fb_last_r && drain_will_empty;
    assign smem_wr_last_pkt = is_last_drain && (!is_multicast || replay_is_last);

    // Completion attr: bar_stride hardcoded to 1.
    // Use mc_write_valid (not mc_write_fire) to break a UNOPTFLAT loop:
    // mc_write_fire = mc_write_valid && smem_bus_if.req_ready, and using
    // it on the request side closes a comb cycle through the downstream
    // LMEM arbiter. Attr is sampled by the receiver only when req_valid
    // && req_ready, so qualifying with req_ready here is redundant.
    wire smem_wr_attr_last = active_notify_smem_done && (
        is_multicast ? (mc_write_valid && is_last_drain) : smem_wr_last_pkt);
    wire [BAR_ADDR_W-1:0] smem_wr_attr_bar = is_multicast
        ? BAR_ADDR_W'(active_bar_addr + (BAR_ADDR_W'(replay_next_idx) << NB_BITS))
        : active_bar_addr;

    // ════════════════════════════════════════════════════════════════════
    // SMEM bus wiring
    // ════════════════════════════════════════════════════════════════════
    assign smem_bus_if.req_valid       = smem_wr_valid;
    assign smem_bus_if.req_data.rw     = 1'b1;
    assign smem_bus_if.req_data.addr   = smem_wr_addr;
    assign smem_bus_if.req_data.data   = fb_word_data;
    assign smem_bus_if.req_data.byteen = fb_word_byteen;
    assign smem_bus_if.req_data.attr   = {smem_wr_attr_last, smem_wr_attr_bar};
    assign smem_bus_if.req_data.tag.uuid  = '0;
    assign smem_bus_if.req_data.tag.value = SMEM_TAG_VALUE_W'(active_core_id) << ENGINE_VALUE_W;
    assign smem_bus_if.rsp_ready       = 1'b0;

    `UNUSED_VAR (smem_bus_if.rsp_valid)
    `UNUSED_VAR (smem_bus_if.rsp_data)
    `UNUSED_VAR (fb_load_oob)

    // ════════════════════════════════════════════════════════════════════
    // Completion tracking
    // ════════════════════════════════════════════════════════════════════
    reg [31:0] wr_count_r;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wr_count_r <= '0;
        end else if (smem_req_fire) begin
            wr_count_r <= wr_count_r + 32'd1;
        end
    end

    assign wr_done_count = wr_count_r;
`ifdef DXA_OOO_DRAIN_ENABLE
    // OoO: last-drained CL is not necessarily the last-issued CL. Done is
    // signalled one cycle after the last release, when pending_size has
    // observed all releases (addr_gen_done is latched once ag_last fires).
    assign transfer_done = transfer_active && addr_gen_done && pending_empty;
    `UNUSED_VAR (smem_wr_last_pkt)
`else
    assign transfer_done = transfer_active && smem_req_fire && smem_wr_last_pkt;
`endif

`ifdef PERF_ENABLE
    reg [31:0] wrp_total_lmem_writes_r;
    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wrp_total_lmem_writes_r <= '0;
        end else if (smem_req_fire) begin
            wrp_total_lmem_writes_r <= wrp_total_lmem_writes_r + 32'd1;
        end
    end
    assign perf_lmem_writes = wrp_total_lmem_writes_r + 32'(smem_req_fire);
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (sw_accept) begin
                $write("DXA_PIPE,%0d,SW_RX,tag=%0d,oob=%0d,last=%0d,path=%s\n",
                    $time, sw_tag, sw_oob, sw_last,
                    sw_defer_path ? "defer" : sw_bypass_path ? "bypass" : "pend");
            end
            if (smem_req_fire) begin
                $write("DXA_PIPE,%0d,SMEM_WR,addr=0x%0h,byteen=0x%0h,last=%0d\n",
                    $time, smem_wr_addr, fb_word_byteen, smem_wr_last_pkt);
            end
        end
    end
`endif

endmodule
